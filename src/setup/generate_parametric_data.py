import numpy as np
import os
import sys
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Clawpack imports
from clawpack import pyclaw
from clawpack import riemann

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def solve_saint_venant_pyclaw(L, T_total, Nx_save, Nt_save, slope, manning, Q_in_func, h_out_val, h_ic_func, u_ic_func):
    """
    Solves 1D Saint-Venant equations using PyClaw.
    
    Returns:
        h_interp: (Nt_save, Nx_save)
        u_interp: (Nt_save, Nx_save)
        t_save: (Nt_save,)
    """
    
    # 1. Domain and Grid
    # Use a finer grid for internal solver to ensure stability and accuracy
    Nx_pyclaw = max(Nx_save * 2, 200) 
    x_lower = 0.0
    x_upper = L
    
    x_dim = pyclaw.Dimension(x_lower, x_upper, Nx_pyclaw, name='x')
    domain = pyclaw.Domain(x_dim)
    
    # 2. State
    num_eqn = 2 # h, hu
    state = pyclaw.State(domain, num_eqn)
    
    # Physics parameters
    g = 9.81
    state.problem_data['grav'] = g
    state.problem_data['slope'] = slope
    state.problem_data['manning'] = manning
    
    # Initial Condition
    xc = state.grid.x.centers
    
    # Evaluate IC functions on grid centers
    h0 = h_ic_func(xc)
    u0 = u_ic_func(xc)
    
    state.q[0, :] = h0
    state.q[1, :] = h0 * u0 # hu
    
    # 3. Solver
    # Use Riemann solver for Shallow Water (Roe)
    rs = riemann.shallow_roe_with_efix_1D
    solver = pyclaw.ClawSolver1D(rs)
    
    solver.limiters = pyclaw.limiters.tvd.vanleer
    
    # Boundary Conditions
    # PyClaw uses 0 for lower, 1 for upper
    solver.bc_lower[0] = pyclaw.BC.custom
    solver.bc_upper[0] = pyclaw.BC.custom
    
    # Custom BC Functions
    def bc_inlet(state, dim, t, qbc, auxbc, num_ghost):
        """
        Inlet BC at x=0.
        Prescribe Q(t) = Q_in_func(t).
        Use characteristic variables to find h.
        """
        # Ghost cells are qbc[:, :num_ghost]
        # Interior cells are qbc[:, num_ghost:]
        
        # We only need to set ghost cells. 
        # For simplicity, we can assume zero-gradient for h (or characteristic BC)
        # and enforce Q. Or use the characteristic invariant.
        
        # Simple approach for subcritical flow:
        # 1. Extrapolate h (zero gradient) -> h_ghost = h_inner
        # 2. Set hu_ghost = Q_in(t)
        
        # Better approach (Characteristic):
        # R- = u - 2c comes from interior.
        # Q_in is given.
        
        # Interior value (first cell)
        h_inner = qbc[0, num_ghost]
        hu_inner = qbc[1, num_ghost]
        u_inner = hu_inner / h_inner
        c_inner = np.sqrt(g * h_inner)
        
        R_minus = u_inner - 2 * c_inner
        
        Q_target = Q_in_func(t)
        
        # Solve for h_bnd: Q/h - 2*sqrt(gh) = R_minus
        # Newton iteration
        h_bnd = h_inner # Initial guess
        for _ in range(5):
            if h_bnd <= 0: h_bnd = 0.01
            f = (Q_target / h_bnd) - 2 * np.sqrt(g * h_bnd) - R_minus
            df = -Q_target / (h_bnd**2) - np.sqrt(g / h_bnd)
            h_bnd = h_bnd - f / df
            
        # Set ghost cells
        # We set all ghost cells to this boundary value
        qbc[0, :num_ghost] = h_bnd
        qbc[1, :num_ghost] = Q_target
        
    def bc_outlet(state, dim, t, qbc, auxbc, num_ghost):
        """
        Outlet BC at x=L.
        Fixed Level h = h_out_val.
        """
        # Interior value (last cell)
        h_inner = qbc[0, -num_ghost-1]
        hu_inner = qbc[1, -num_ghost-1]
        u_inner = hu_inner / h_inner
        c_inner = np.sqrt(g * h_inner)
        
        # Characteristic R+ = u + 2c comes from interior
        R_plus = u_inner + 2 * c_inner
        
        h_target = h_out_val
        
        # u_bnd = R_plus - 2*sqrt(g * h_target)
        u_bnd = R_plus - 2 * np.sqrt(g * h_target)
        hu_bnd = h_target * u_bnd
        
        # Set ghost cells
        qbc[0, -num_ghost:] = h_target
        qbc[1, -num_ghost:] = hu_bnd

    solver.user_bc_lower = bc_inlet
    solver.user_bc_upper = bc_outlet
    
    # Source Term (Manning Friction + Slope)
    def step_source(solver, state, dt):
        """
        Update state.q with source terms using Euler or Semi-Implicit step.
        S = g * h * (S0 - Sf)
        Sf = n^2 * u * |u| / h^(4/3)
        Momentum equation source: g * h * (S0 - Sf)
        """
        h = state.q[0, :]
        hu = state.q[1, :]
        
        # Avoid divide by zero
        h_safe = np.maximum(h, 1e-3)
        u = hu / h_safe
        
        Sf = (manning**2 * u * np.abs(u)) / (h_safe**(4/3))
        
        # Source contribution to momentum
        # d(hu)/dt = ... + g * h * (S0 - Sf)
        S_mom = g * h_safe * (slope - Sf)
        
        state.q[1, :] += dt * S_mom

    solver.step_source = step_source
    solver.source_split = 1 # Strang splitting
    
    # 4. Controller
    controller = pyclaw.Controller()
    controller.solution = pyclaw.Solution(state, domain)
    controller.solver = solver
    controller.tfinal = T_total
    # To capture dynamics properly and allow interpolation, we output frequent frames
    # We aim for Nt_save, but PyClaw might need more internal steps.
    # Let's just output at least Nt_save frames.
    controller.num_output_times = Nt_save 
    
    # Keep frames in memory
    controller.keep_copy = True
    
    # Silence output
    controller.verbosity = 0
    
    # 5. Run
    status = controller.run()
    
    # 6. Extract and Interpolate Data
    # controller.frames is a list of Solution objects
    # We want to interpolate to (t_save, x_save)
    
    t_save = np.linspace(0, T_total, Nt_save)
    x_save = np.linspace(0, L, Nx_save)
    
    # Collect data from frames
    t_frames = []
    h_frames = []
    u_frames = []
    
    for frame in controller.frames:
        t_frames.append(frame.t)
        # Interpolate spatially to x_save
        xc_frame = frame.states[0].grid.x.centers
        h_frame = frame.states[0].q[0, :]
        hu_frame = frame.states[0].q[1, :]
        u_frame = hu_frame / np.maximum(h_frame, 1e-3)
        
        # Spatial interpolation
        f_h = interp1d(xc_frame, h_frame, kind='linear', fill_value="extrapolate")
        f_u = interp1d(xc_frame, u_frame, kind='linear', fill_value="extrapolate")
        
        h_frames.append(f_h(x_save))
        u_frames.append(f_u(x_save))
        
    t_frames = np.array(t_frames)
    h_frames = np.array(h_frames) # (N_frames, Nx_save)
    u_frames = np.array(u_frames)
    
    # Temporal interpolation to t_save
    # PyClaw output times might be exactly what we asked for, but let's ensure alignment
    f_h_t = interp1d(t_frames, h_frames, axis=0, kind='linear', fill_value="extrapolate")
    f_u_t = interp1d(t_frames, u_frames, axis=0, kind='linear', fill_value="extrapolate")
    
    h_interp = f_h_t(t_save)
    u_interp = f_u_t(t_save)
    
    return h_interp, u_interp, t_save

def generate_parametric_dataset(num_samples=50):
    print(f"Generating {num_samples} parametric samples using PyClaw...")
    
    # Target Grid
    L = 10000.0
    T_total = 14400.0 
    Nx = 200
    Nt = 241 # dt = 60s
    
    print(f"Grid: Nx={Nx}, Nt={Nt}, dt={T_total/(Nt-1):.2f}s")
    
    slope = 0.001
    manning = 0.03
    
    # Containers
    bc_data = np.zeros((num_samples, Nt, 4), dtype=np.float32)
    field_data = np.zeros((num_samples, Nt, Nx, 2), dtype=np.float32)
    
    for i in tqdm(range(num_samples), desc="Simulating"):
        # Parameters
        h0_base = np.random.uniform(4.0, 6.0)
        Q0_base = np.random.uniform(15.0, 25.0)
        
        # Hydrograph
        has_pulse = np.random.rand() > 0.2
        if has_pulse:
            amp = np.random.uniform(10.0, 40.0)
            t_peak = np.random.uniform(0.2 * T_total, 0.6 * T_total)
            sigma = np.random.uniform(0.05 * T_total, 0.15 * T_total)
        else:
            amp = 0; t_peak = 0; sigma = 1.0
            
        def Q_in_func(t):
            return Q0_base + amp * np.exp(- (t - t_peak)**2 / (2 * sigma**2))
        
        h_out_val = h0_base
        
        # Initial Condition Functions
        # Create a closure to capture randomized noise
        noise_h = np.random.normal(0, 0.05, 1000) # Pre-generate noise buffer
        noise_u = np.random.normal(0, 0.1, 1000)
        
        def h_ic_func(x):
            # Map x to noise buffer indices
            idx = (x / L * 999).astype(int)
            return h0_base + noise_h[idx]
            
        def u_ic_func(x):
            idx = (x / L * 999).astype(int)
            return (Q0_base / h0_base) + noise_u[idx]
            
        # Solve
        h_res, u_res, t_res = solve_saint_venant_pyclaw(
            L, T_total, Nx, Nt, slope, manning,
            Q_in_func, h_out_val, h_ic_func, u_ic_func
        )
        
        # Store
        field_data[i, :, :, 0] = h_res
        field_data[i, :, :, 1] = u_res
        
        # BCs
        bc_data[i, :, 0] = h_res[:, 0]
        bc_data[i, :, 1] = u_res[:, 0]
        bc_data[i, :, 2] = h_res[:, -1]
        bc_data[i, :, 3] = u_res[:, -1]
        
    # Save
    save_dir = os.path.join("data", "SaintVenant1D", "parametric_batch")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    np.save(os.path.join(save_dir, "bc_data.npy"), bc_data)
    np.save(os.path.join(save_dir, "field_data.npy"), field_data)
    
    x_grid = np.linspace(0, L, Nx)
    np.save(os.path.join(save_dir, "x_grid.npy"), x_grid)
    np.save(os.path.join(save_dir, "t_grid.npy"), t_res)
    
    print(f"Dataset saved to {save_dir}")
    
    # Verify Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_res, bc_data[0, :, 1] * bc_data[0, :, 0], label='Inlet Q')
    plt.title("Sample 0: Hydrograph")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.imshow(field_data[0, :, :, 0], aspect='auto', origin='lower', extent=[0, L, 0, T_total])
    plt.colorbar(label='h (m)')
    plt.title("Sample 0: Depth")
    plt.savefig(os.path.join(save_dir, "pyclaw_preview.png"))
    print("Preview saved.")

if __name__ == "__main__":
    generate_parametric_dataset(50)
