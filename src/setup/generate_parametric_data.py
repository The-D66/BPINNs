import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path to import config if needed, though we will be explicit here
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def solve_saint_venant_maccormack(L, T_total, Nx, Nt, slope, manning, Q_in_func, h_out_val, h_ic, Q_ic):
    """
    Solves 1D Saint-Venant equations using MacCormack scheme for a SINGLE case.
    
    Args:
        L: Length of domain (m)
        T_total: Total simulation time (s)
        Nx: Number of spatial points
        Nt: Number of time steps
        slope: Bed slope
        manning: Manning's roughness coefficient
        Q_in_func: Function Q_in(t) returning scalar
        h_out_val: Scalar value for outlet h (fixed)
        h_ic: Initial condition array for h (shape Nx)
        Q_ic: Initial condition array for Q (shape Nx)
        
    Returns:
        h_field: (Nt, Nx) array
        u_field: (Nt, Nx) array
        t_grid: (Nt,) array
    """
    dx = L / (Nx - 1)
    dt = T_total / (Nt - 1)
    g = 9.81
    
    # State variables
    h = h_ic.copy()
    Q = Q_ic.copy()
    
    # Storage
    h_field = np.zeros((Nt, Nx))
    u_field = np.zeros((Nt, Nx))
    
    h_field[0, :] = h
    u_field[0, :] = Q / np.maximum(h, 1e-3)
    
    t_grid = np.linspace(0, T_total, Nt)
    
    # Helper for Source Term
    def get_source_term(h_curr, Q_curr):
        h_safe = np.maximum(h_curr, 1e-3)
        u_curr = Q_curr / h_safe
        Sf = (manning**2 * u_curr * np.abs(u_curr)) / (h_safe**(4/3))
        return g * h_safe * (slope - Sf)

    for n in range(1, Nt):
        t_curr = t_grid[n]
        
        # --- Predictor Step ---
        h_p = h.copy()
        Q_p = Q.copy()
        
        F1 = Q
        F2 = (Q**2 / np.maximum(h, 1e-3)) + 0.5 * g * h**2
        S = get_source_term(h, Q)
        
        # Forward difference
        h_p[0:-1] = h[0:-1] - (dt/dx) * (F1[1:] - F1[0:-1])
        Q_p[0:-1] = Q[0:-1] - (dt/dx) * (F2[1:] - F2[0:-1]) + dt * S[0:-1]
        
        # --- Corrector Step ---
        h_p = np.maximum(h_p, 1e-3)
        F1_p = Q_p
        F2_p = (Q_p**2 / np.maximum(h_p, 1e-3)) + 0.5 * g * h_p**2
        S_p = get_source_term(h_p, Q_p)
        
        h_new = np.zeros_like(h)
        Q_new = np.zeros_like(Q)
        
        # Backward difference
        h_new[1:] = 0.5 * (h[1:] + h_p[1:] - (dt/dx) * (F1_p[1:] - F1_p[0:-1]))
        Q_new[1:] = 0.5 * (Q[1:] + Q_p[1:] - (dt/dx) * (F2_p[1:] - F2_p[0:-1]) + dt * S_p[1:])
        
        # --- Boundary Conditions ---
        # Inlet (x=0): Q prescribed, h computed via characteristics
        Q_new[0] = Q_in_func(t_curr)
        
        # Characteristic invariant approximation at inlet
        # R- = u - 2c. From grid 1 to 0.
        u_1 = Q_new[1]/h_new[1]
        c_1 = np.sqrt(g*h_new[1])
        R_minus = u_1 - 2*c_1
        
        # Solve for h_0: Q_in/h_0 - 2*sqrt(g*h_0) = R_minus
        # Newton-Raphson
        h_guess = h_new[1]
        for _ in range(3):
            f = (Q_new[0]/h_guess) - 2*np.sqrt(g*h_guess) - R_minus
            df = -Q_new[0]/(h_guess**2) - np.sqrt(g/h_guess)
            h_guess = h_guess - f/df
        h_new[0] = np.maximum(h_guess, 0.1)
        
        # Outlet (x=L): h prescribed (fixed weir/lake), Q computed via characteristics
        h_new[-1] = h_out_val
        
        # Characteristic invariant approximation at outlet
        # R+ = u + 2c. From grid N-2 to N-1.
        u_N2 = Q_new[-2]/h_new[-2]
        c_N2 = np.sqrt(g*h_new[-2])
        R_plus = u_N2 + 2*c_N2
        
        # u_out = R_plus - 2*sqrt(g*h_out)
        u_out = R_plus - 2*np.sqrt(g*h_new[-1])
        Q_new[-1] = u_out * h_new[-1]
        
        # Update
        h = h_new
        Q = Q_new
        
        h_field[n, :] = h
        u_field[n, :] = Q / np.maximum(h, 1e-3)
        
    return h_field, u_field, t_grid

def generate_parametric_dataset(num_samples=50):
    print(f"Generating {num_samples} parametric samples for Operator Learning...")
    
    # Fixed Physics Parameters
    L = 10000.0
    T_total = 14400.0 # 4 hours
    Nx = 128 # Spatial resolution for saved data
    Nt = 100 # Temporal resolution for saved data (downsample if solver uses finer steps)
    
    # Solver Grid (Fine for stability)
    Nx_solver = 200
    
    # CFL Check
    h_max_est = 10.0
    u_max_est = 5.0
    c_max = np.sqrt(9.81 * h_max_est)
    dx = L / (Nx_solver - 1)
    dt_cfl = 0.5 * dx / (u_max_est + c_max)
    Nt_solver = int(T_total / dt_cfl) + 1
    print(f"Solver Grid: Nx={Nx_solver}, Nt={Nt_solver}, dt={dt_cfl:.3f}s")
    
    slope = 0.001
    manning = 0.03
    
    # Data Containers
    # BC: (N, Nt, 4) -> [h_in, u_in, h_out, u_out]
    bc_data = np.zeros((num_samples, Nt, 4), dtype=np.float32)
    # IC: (N, Nx, 2) -> [h_0, u_0] (using saved grid Nx)
    ic_data = np.zeros((num_samples, Nx, 2), dtype=np.float32)
    # Field: (N, Nt, Nx, 2) -> [h, u]
    field_data = np.zeros((num_samples, Nt, Nx, 2), dtype=np.float32)
    
    # Sampling Loop
    for i in tqdm(range(num_samples), desc="Simulating"):
        # 1. Randomize Parameters
        
        # Base Flow
        h0_base = np.random.uniform(4.0, 6.0)
        Q0_base = np.random.uniform(15.0, 25.0)
        
        # Inflow Hydrograph (Gaussian Pulse)
        # Q_in(t) = Q0 + Amp * exp(- (t - t_peak)^2 / (2 * sigma^2) )
        has_pulse = np.random.rand() > 0.2 # 80% chance of flood wave
        if has_pulse:
            amp = np.random.uniform(10.0, 40.0) # Peak increase
            t_peak = np.random.uniform(0.2 * T_total, 0.6 * T_total)
            sigma = np.random.uniform(0.05 * T_total, 0.15 * T_total)
        else:
            amp = 0; t_peak = 0; sigma = 1.0
            
        def Q_in_func(t):
            return Q0_base + amp * np.exp(- (t - t_peak)**2 / (2 * sigma**2))
        
        # Outlet (Fixed Level for now, could be randomized)
        h_out_val = h0_base # Simple matching boundary
        
        # Initial Condition (Steady state approx + small noise)
        # Start with uniform
        h_ic_solver = np.ones(Nx_solver) * h0_base
        Q_ic_solver = np.ones(Nx_solver) * Q0_base
        # Add spatial noise (simulating measurement error or non-steady start)
        h_ic_solver += np.random.normal(0, 0.05, Nx_solver)
        Q_ic_solver += np.random.normal(0, 0.1, Nx_solver)
        
        # 2. Run Solver
        h_res, u_res, t_solver = solve_saint_venant_maccormack(
            L, T_total, Nx_solver, Nt_solver, slope, manning, 
            Q_in_func, h_out_val, h_ic_solver, Q_ic_solver
        )
        
        # 3. Downsample/Interpolate to saved grid (Nx, Nt)
        # Time downsampling
        t_indices = np.linspace(0, Nt_solver-1, Nt).astype(int)
        h_time_down = h_res[t_indices, :]
        u_time_down = u_res[t_indices, :]
        
        # Spatial downsampling
        x_indices = np.linspace(0, Nx_solver-1, Nx).astype(int)
        h_final = h_time_down[:, x_indices]
        u_final = u_time_down[:, x_indices]
        
        # 4. Store Data
        
        # Field
        field_data[i, :, :, 0] = h_final
        field_data[i, :, :, 1] = u_final
        
        # IC (t=0 snapshot)
        ic_data[i, :, 0] = h_final[0, :]
        ic_data[i, :, 1] = u_final[0, :]
        
        # BC (x=0 and x=L time series)
        # Upstream [h(0,t), u(0,t)]
        bc_data[i, :, 0] = h_final[:, 0]
        bc_data[i, :, 1] = u_final[:, 0] # Note: u is calculated, Q_in was prescribed. u=Q/h.
        # Downstream [h(L,t), u(L,t)]
        bc_data[i, :, 2] = h_final[:, -1]
        bc_data[i, :, 3] = u_final[:, -1]
        
    # 5. Save to Disk
    save_dir = os.path.join("../data/SaintVenant1D", "parametric_batch")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    np.save(os.path.join(save_dir, "bc_data.npy"), bc_data)
    np.save(os.path.join(save_dir, "ic_data.npy"), ic_data)
    np.save(os.path.join(save_dir, "field_data.npy"), field_data)
    
    # Save Coordinates
    x_grid = np.linspace(0, L, Nx)
    t_grid = np.linspace(0, T_total, Nt)
    np.save(os.path.join(save_dir, "x_grid.npy"), x_grid)
    np.save(os.path.join(save_dir, "t_grid.npy"), t_grid)
    
    print(f"Dataset saved to {save_dir}")
    print(f"Shapes: BC={bc_data.shape}, IC={ic_data.shape}, Field={field_data.shape}")
    
    # Plot one sample to verify
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_grid, bc_data[0, :, 1] * bc_data[0, :, 0], label='Inlet Q')
    plt.plot(t_grid, bc_data[0, :, 3] * bc_data[0, :, 2], label='Outlet Q')
    plt.title(f"Sample 0: Hydrographs")
    plt.xlabel("Time (s)")
    plt.ylabel("Q (m3/s)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.imshow(field_data[0, :, :, 0], aspect='auto', origin='lower', 
               extent=[0, L, 0, T_total])
    plt.colorbar(label='h (m)')
    plt.title("Sample 0: Water Depth Field")
    plt.xlabel("x (m)")
    plt.ylabel("t (s)")
    
    plot_path = os.path.join(save_dir, "sample_0_preview.png")
    plt.savefig(plot_path)
    print(f"Preview plot saved to {plot_path}")

if __name__ == "__main__":
    generate_parametric_dataset(num_samples=50) # Small batch for testing
