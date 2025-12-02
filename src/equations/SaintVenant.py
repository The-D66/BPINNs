from .Equation import Equation
from .Operators import Operators
import tensorflow as tf


class SaintVenant(Equation):
  """
    1D Saint-Venant (Shallow Water) equations implementation
    """
  def __init__(self, par):
    super().__init__(par)
    self.g = 9.81
    # Physical scales (passed from config/physics)
    self.L = par.physics.get("length", 1000.0)
    self.T = par.physics.get("time", 3600.0)
    # Get S0 and n_manning from par.physics, with defaults
    self.S0 = par.physics.get("slope", 0.0)
    self.n_manning = par.physics.get("manning", 0.0)
    self.viscosity = par.physics.get("viscosity", 0.0) # Artificial Viscosity

    # Optimization: Pre-compute constants to avoid re-calculation in the loop
    self.inv_L = 1.0 / self.L
    self.inv_T = 1.0 / self.T
    self.inv_L_sq = self.inv_L**2 # For second derivative
    self.n_manning_sq = self.n_manning**2
    self.g_S0 = self.g * self.S0

  @property
  def norm_coeff(self):
    return self.norm

  @norm_coeff.setter
  def norm_coeff(self, norm):
    # norm["sol_mean"] is numpy array [mean_h, mean_u]
    # We expect sol to have 2 components.
    # Check if scalars or arrays

    # Helper to safe cast
    def to_tf(val):
      return tf.cast(val, tf.float32)

    self.norm["h_mean"] = to_tf(norm["sol_mean"][0])
    self.norm["u_mean"] = to_tf(norm["sol_mean"][1])
    self.norm["h_std"] = to_tf(norm["sol_std"][0])
    self.norm["u_std"] = to_tf(norm["sol_std"][1])

  def comp_residual(self, inputs, out_sol, out_par, tape):
    # Handle Operator Learning inputs (list: [bc, ic, query])
    if isinstance(inputs, (list, tuple)):
        inputs_for_grad = inputs[-1] # Query points [x, t]
    else:
        inputs_for_grad = inputs

    # Unpack solution: out_sol should be [h_norm, u_norm] (normalized)
    # inputs should be [x_norm, t_norm] (normalized [0,1])

    sol_list = Operators.tf_unpack(out_sol)
    h_norm = sol_list[0]  # Normalized Water depth
    u_norm = sol_list[1]  # Normalized Velocity

    # Retrieve normalization stats
    h_mu, h_sigma = self.norm["h_mean"], self.norm["h_std"]
    u_mu, u_sigma = self.norm["u_mean"], self.norm["u_std"]

    # Denormalize variables to Physical Units
    h = h_norm * h_sigma + h_mu
    u = u_norm * u_sigma + u_mu

    # Gradients Calculation
    # gradient_scalar returns [d/dx_norm, d/dt_norm]
    grad_h_norm = Operators.gradient_scalar(tape, h_norm, inputs_for_grad)
    grad_u_norm = Operators.gradient_scalar(tape, u_norm, inputs_for_grad)

    # Unpack normalized gradients
    # grad is (N, 2). col 0 is d/dx*, col 1 is d/dt*
    h_x_norm = grad_h_norm[:, 0:1]
    h_t_norm = grad_h_norm[:, 1:2]
    u_x_norm = grad_u_norm[:, 0:1]
    u_t_norm = grad_u_norm[:, 1:2]

    # Artificial Viscosity: Need u_xx
    # Only calculate if viscosity > 0 to save compute
    if self.viscosity > 0:
        grad_u_x_norm = Operators.gradient_scalar(tape, u_x_norm, inputs_for_grad)
        u_xx_norm = grad_u_x_norm[:, 0:1]
        u_xx = u_xx_norm * u_sigma * self.inv_L_sq
    else:
        u_xx = 0.0

    # Convert Gradients to Physical Units
    # dH/dX = (dH/dh_norm) * (dh_norm/dx_norm) * (dx_norm/dX)
    # dH/dX = sigma_h * h_x_norm * (1/L)

    # Optimization: Use pre-computed inverse constants
    h_x = h_x_norm * h_sigma * self.inv_L
    h_t = h_t_norm * h_sigma * self.inv_T
    u_x = u_x_norm * u_sigma * self.inv_L
    u_t = u_t_norm * u_sigma * self.inv_T

    # Physical Equations
    # 1. Continuity Equation: h_t + (hu)_x = 0  => h_t + h_x u + h u_x = 0
    lhs_cont = h_t + (h_x * u + h * u_x)

    # 2. Momentum Equation: u_t + u u_x + g h_x + Sf - S0 = 0 (+ viscosity term)
    # With viscosity: u_t + u u_x + g h_x + g(Sf - S0) - nu * u_xx = 0

    # Friction term (Manning): Sf = n^2 * u * |u| / h^(4/3)
    if self.n_manning > 0:
      # Use softplus for smoother gradients
      h_safe = tf.math.softplus(h - 0.5) + 0.5
      # Optimization: Use pre-computed n_manning_sq
      Sf = (self.n_manning_sq * u * tf.abs(u)) / (h_safe**(4 / 3))

      # Optimization: Combine constants g * (S0 - Sf) -> g_S0 - g*Sf
      term_forcing = self.g_S0 - self.g * Sf
    else:
      term_forcing = self.g_S0

    # Corrected momentum equation residual
    # Added viscosity term: - nu * u_xx
    lhs_mom = u_t + u * u_x + self.g * h_x - term_forcing - self.viscosity * u_xx

    # Return both residuals concatenated
    return tf.concat([lhs_cont, lhs_mom], axis=1)
