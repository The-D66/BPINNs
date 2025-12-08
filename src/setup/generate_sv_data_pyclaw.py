import numpy as np
import os
from clawpack import pyclaw
from clawpack import riemann
import matplotlib.pyplot as plt

def friction_source(solver, state, dt):
    """
    Source term for Manning friction and Bed Slope.
    Momentum equation source: g * h * (S0 - Sf)
    Sf = n^2 * u * |u| / h^(4/3)
    """
    g = 9.81
    n_manning = state.problem_data['manning']
    slope = state.problem_data['slope']
    
    q = state.q
    h = q[0, :]
    hu = q[1, :]
    
    # Regularize h to avoid division by zero
    h_safe = np.maximum(h, 1e-3)
    u = hu / h_safe
    
    # Friction Slope
    Sf = (n_manning**2 * u * np.abs(u)) / (h_safe**(4.0/3.0))
    
    # Source term (Gravity force component along slope - Friction)
    # Applied to momentum (q[1])
    # Source = g * h * (S0 - Sf)
    S = g * h * (slope - Sf)
    
    # Update momentum
    q[1, :] += dt * S

def q_bc_lower(state, dim, t, qbc, auxbc, num_ghost):
    """
    Upstream Boundary Condition (x=0)
    Prescribe Q(t), extrapolate h.
    """
    # Physics parameters
    Q0 = state.problem_data['Q0']
    T_warmup = state.problem_data['T_warmup']
    
    # Calculate target Q_in based on time t
    # Note: t is current simulation time
    if t < T_warmup:
        Q_in = Q0
    elif t <= T_warmup + 3600.0:
        # Ramp down over 1 hour after warmup
        ratio = (t - T_warmup) / 3600.0
        Q_in = Q0 * (1.0 - 0.5 * ratio)
    else:
        Q_in = 0.5 * Q0
        
    # For ghost cells (left of x=0)
    # We need to set qbc (which includes ghost cells)
    # dim.lower is the index of the first physical cell
    
    # Zero-order extrapolation for h
    h_internal = qbc[0, num_ghost]
    
    for i in range(num_ghost):
        # Set h in ghost cells
        qbc[0, i] = h_internal
        # Set hu based on Q_in
        qbc[1, i] = Q_in

def q_bc_upper(state, dim, t, qbc, auxbc, num_ghost):
    """
    Downstream Boundary Condition (x=L)
    Prescribe h, extrapolate Q/u.
    """
    h0 = state.problem_data['h0']
    
    # Zero-order extrapolation for momentum hu
    hu_internal = qbc[1, -num_ghost-1]
    
    for i in range(num_ghost):
        # Index from end: -1, -2 ...
        idx = -1 - i
        qbc[0, idx] = h0
        qbc[1, idx] = hu_internal

def generate_pyclaw_data():
    # Configuration
    L = 10000.0
    T_warmup = 3600.0
    T_sim_pinn = 7200.0
    T_total = T_warmup + T_sim_pinn 
    
    Nx = 200
    g = 9.81
    h0 = 5.78
    Q0 = 20.0
    slope = 0.001
    manning = 0.03
    
    # Riemann Solver
    rs = riemann.shallow_1D_py.shallow_fwave_1d
    solver = pyclaw.ClawSolver1D(rs)
    solver.num_eqn = 2
    solver.num_waves = 2
    solver.fwave = True
    solver.kernel_language = 'Python'
    
    # Limiters (TVD)
    solver.limiters = pyclaw.limiters.tvd.vanleer
    
    # Boundary Conditions
    solver.bc_lower[0] = pyclaw.BC.custom
    solver.user_bc_lower = q_bc_lower
    solver.bc_upper[0] = pyclaw.BC.extrap
    # solver.user_bc_upper removed as we use extrap
    
    solver.aux_bc_lower[0] = pyclaw.BC.extrap
    solver.aux_bc_upper[0] = pyclaw.BC.extrap
    
    # Source Term
    solver.step_source = friction_source
    solver.source_split = 1 # Godunov splitting (first order)
    
    # Domain
    x = pyclaw.Dimension(0.0, L, Nx, name='x')
    domain = pyclaw.Domain(x)
    
    # Initial State
    state = pyclaw.State(domain, solver.num_eqn, num_aux=1)
    state.problem_data['grav'] = g
    state.problem_data['manning'] = manning
    state.problem_data['slope'] = slope
    state.problem_data['h0'] = h0
    state.problem_data['Q0'] = Q0
    state.problem_data['T_warmup'] = T_warmup
    state.problem_data['dry_tolerance'] = 1e-3
    state.problem_data['sea_level'] = 0.0
    
    # Initialize bathymetry (aux[0])
    xc = state.grid.x.centers
    state.aux[0, :] = -slope * xc
    
    # Initial Condition: Steady uniform flow
    xc = state.grid.x.centers
    state.q[0, :] = h0
    state.q[1, :] = Q0 # Momentum hu = Q
    
    # Controller
    claw = pyclaw.Controller()
    claw.tfinal = T_total
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    
    # Output settings
    # We want to save frames similar to previous script
    # Previous script saved roughly every 2.2s (dt) or 10s.
    # Let's save 200 frames to get high temporal resolution
    claw.num_output_times = 200
    claw.output_format = None # Disable file output, we'll collect in memory
    claw.keep_copy = True # Keep solutions in memory
    
    print("Running PyClaw simulation...")
    status = claw.run()
    print(f"Simulation finished with status: {status}")
    
    # Extract Data
    # claw.frames is a list of Solution objects
    frames = claw.frames
    n_save = len(frames)
    
    h_history = np.zeros((n_save, Nx))
    u_history = np.zeros((n_save, Nx))
    t_history = np.zeros(n_save)
    x_history = xc # Grid centers
    
    for i, frame in enumerate(frames):
        h = frame.q[0, :]
        hu = frame.q[1, :]
        h_safe = np.maximum(h, 1e-3)
        u = hu / h_safe
        
        h_history[i, :] = h
        u_history[i, :] = u
        t_history[i] = frame.t
        
    # Save Data
    os.makedirs("data_raw", exist_ok=True)
    np.save("data_raw/h_history.npy", h_history)
    np.save("data_raw/u_history.npy", u_history)
    np.save("data_raw/x_grid.npy", x_history)
    np.save("data_raw/t_grid.npy", t_history)
    np.save("data_raw/config.npy", {"L": L, "T_warmup": T_warmup, "T_sim": T_sim_pinn, "T_total_sim": T_total, "S0": slope, "n_manning": manning})
    
    print(f"Saved {n_save} frames to data_raw/")
    
    # Plot Check
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(h_history, aspect='auto', extent=[0, L, T_total, 0], cmap='viridis')
    plt.colorbar(label='h (m)')
    plt.title("PyClaw: Water Depth")
    plt.axhline(y=T_warmup, color='r', linestyle='--')
    
    plt.subplot(2, 1, 2)
    plt.imshow(u_history, aspect='auto', extent=[0, L, T_total, 0], cmap='viridis')
    plt.colorbar(label='u (m/s)')
    plt.title("PyClaw: Velocity")
    plt.axhline(y=T_warmup, color='r', linestyle='--')
    
    plt.savefig("data_raw/pyclaw_check.png")
    print("Check plot saved.")

if __name__ == "__main__":
    generate_pyclaw_data()
