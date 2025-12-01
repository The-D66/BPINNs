from .Equation import Equation
from .Operators import Operators
import tensorflow as tf

class SaintVenantHamiltonian(Equation):
    """
    Hamiltonian Neural Network (HNN) style implementation of Saint-Venant equations.
    
    Instead of hardcoding the PDE terms, we define the Hamiltonian (Energy) Density
    and use Automatic Differentiation (AD) to compute the variational derivatives
    and their spatial gradients, mimicking the symplectic structure.
    
    Hamiltonian Density H(h, u) = K(h, u) + P(h)
    K = 0.5 * h * u^2  (Kinetic Energy density)
    P = 0.5 * g * h^2 + g * h * z_b (Potential Energy density, including bed elevation)
    
    Dynamics:
    h_t = - d/dx ( dH/du )   => Mass Conservation
    u_t = - d/dx ( dH/dh ) + NonConservativeForces   => Momentum Conservation
    
    Where:
    dH/du = h * u (Momentum density)
    dH/dh = 0.5 * u^2 + g * h + g * z_b (Bernoulli Head)
    """
    def __init__(self, par):
        super().__init__(par)
        self.g = 9.81
        # Physical scales
        self.L = par.physics.get("length", 1000.0)
        self.T = par.physics.get("time", 3600.0)
        self.S0 = par.physics.get("slope", 0.0)
        self.n_manning = par.physics.get("manning", 0.0)
        self.viscosity = par.physics.get("viscosity", 0.0)

        # Optimization: Pre-compute constants
        self.inv_L = 1.0 / self.L
        self.inv_T = 1.0 / self.T
        self.inv_L_sq = self.inv_L**2
        self.n_manning_sq = self.n_manning**2
        
        # Bed slope z_b = -S0 * x. 
        # Note: We handle S0 via the Potential Energy term g*h*z_b
        
    @property
    def norm_coeff(self):
        return self.norm

    @norm_coeff.setter
    def norm_coeff(self, norm):
        def to_tf(val):
            return tf.cast(val, tf.float32)

        self.norm["h_mean"] = to_tf(norm["sol_mean"][0])
        self.norm["u_mean"] = to_tf(norm["sol_mean"][1])
        self.norm["h_std"] = to_tf(norm["sol_std"][0])
        self.norm["u_std"] = to_tf(norm["sol_std"][1])

    def comp_residual(self, inputs, out_sol, out_par, tape):
        # inputs: (N, 2) [x_norm, t_norm]
        
        # 1. Recover Physical Variables
        sol_list = Operators.tf_unpack(out_sol)
        h_norm = sol_list[0]
        u_norm = sol_list[1]

        h_mu, h_sigma = self.norm["h_mean"], self.norm["h_std"]
        u_mu, u_sigma = self.norm["u_mean"], self.norm["u_std"]

        h = h_norm * h_sigma + h_mu
        u = u_norm * u_sigma + u_mu
        
        # Recover Physical Coordinate x for potential energy
        x_norm = inputs[:, 0:1]
        x_phys = x_norm * self.L

        # 2. Compute Time Derivatives (LHS)
        # We use the main tape to get gradients w.r.t inputs (t_norm)
        grad_h_norm = Operators.gradient_scalar(tape, h_norm, inputs)
        grad_u_norm = Operators.gradient_scalar(tape, u_norm, inputs)
        
        h_t = grad_h_norm[:, 1:2] * h_sigma * self.inv_T
        u_t = grad_u_norm[:, 1:2] * u_sigma * self.inv_T
        
        # 3. Hamiltonian Autograd (The HNN Core)
        # We use a nested GradientTape to differentiate Energy w.r.t state variables (h, u).
        # This mimics finding the variational derivatives of the Hamiltonian functional.
        
        with tf.GradientTape(persistent=True) as hnn_tape:
            hnn_tape.watch([h, u])
            
            # Kinetic Energy Density
            # K = 1/2 * h * u^2
            K = 0.5 * h * u**2
            
            # Potential Energy Density
            # P = 1/2 * g * h^2 + g * h * z_b
            # z_b = -S0 * x
            z_b = -self.S0 * x_phys
            P = 0.5 * self.g * h**2 + self.g * h * z_b
            
            # Hamiltonian Density
            H = K + P
            
        # Compute Variational Derivatives (Gradients of H w.r.t state)
        # dH/du = h * u  (Momentum density)
        # dH/dh = 0.5 * u^2 + g * h + g * z_b (Bernoulli Head)
        dH_du = hnn_tape.gradient(H, u)
        dH_dh = hnn_tape.gradient(H, h)
        
        # 4. Compute Spatial Fluxes via AD (Symplectic Structure)
        # We need d/dx of the variational derivatives.
        # Since dH_du and dH_dh depend on h, u (and x), which depend on inputs,
        # we can use the main 'tape' to differentiate them w.r.t inputs.
        
        # Gradients w.r.t inputs [x_norm, t_norm]
        grad_dH_du = Operators.gradient_scalar(tape, dH_du, inputs)
        grad_dH_dh = Operators.gradient_scalar(tape, dH_dh, inputs)
        
        # Extract x-derivative and scale to physical units
        dH_du_x = grad_dH_du[:, 0:1] * self.inv_L
        dH_dh_x = grad_dH_dh[:, 0:1] * self.inv_L
        
        # 5. Assemble Residuals
        
        # Mass Conservation: h_t + (dH/du)_x = 0
        # (dH/du)_x matches d(hu)/dx
        res_mass = h_t + dH_du_x
        
        # Momentum Conservation: u_t + (dH/dh)_x = NonConservative
        # (dH/dh)_x matches d(0.5u^2 + gh - gS0x)/dx = u u_x + g h_x - g S0
        
        # Non-Conservative Forces (Friction & Viscosity)
        # These are not derived from H, so we add them explicitly.
        
        # Manning Friction
        if self.n_manning > 0:
            h_safe = tf.math.softplus(h - 0.5) + 0.5
            Sf = (self.n_manning_sq * u * tf.abs(u)) / (h_safe**(4 / 3))
            force_friction = -self.g * Sf # Retarding force
        else:
            force_friction = 0.0
            
        # Artificial Viscosity
        if self.viscosity > 0:
             # Need u_xx. We already have u_norm gradient.
             grad_u_x_norm = Operators.gradient_scalar(tape, grad_u_norm[:, 0:1], inputs)
             u_xx = grad_u_x_norm[:, 0:1] * u_sigma * self.inv_L_sq
             force_visc = self.viscosity * u_xx
        else:
             force_visc = 0.0
             
        # Total Momentum Residual
        # u_t + (Bernoulli)_x = Forces
        res_mom = u_t + dH_dh_x - (force_friction + force_visc)

        return tf.concat([res_mass, res_mom], axis=1)