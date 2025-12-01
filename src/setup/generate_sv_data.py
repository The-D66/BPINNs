import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.config.saint_venant import SaintVenant1D_simple

def generate_saint_venant_data():
    # Load config from the same object used by PINN
    config_obj = SaintVenant1D_simple()
    physics = config_obj.physics
    
    # Parameters from config
    L = physics["length"]
    T_sim_pinn = physics["time"] # PINN training time
    slope = physics["slope"]
    manning = physics["manning"]
    h0 = physics["h0"]
    Q0 = physics["Q0"]
    
    T_warmup = 3600.0 # 1 hour warmup
    T_total_sim = 18000.0 # 5 hours total simulation
    T_sim_pinn = T_total_sim - T_warmup # 4 hours for PINN training
    
    Nx = 200  # Spatial resolution for solver
    dx = L / (Nx - 1)
    g = 9.81
    
    # Initial Conditions
    u0 = Q0 / h0
    
    # CFL condition (using initial conditions for conservative dt)
    dt = 0.5 * dx / (u0 + np.sqrt(g * h0))
    Nt = int(np.ceil(T_total_sim / dt))
    
    print(f"Solver config: L={L}m, T_total_sim={T_total_sim}s (Warmup={T_warmup}s)")
    print(f"PINN training time: {T_sim_pinn}s")
    print(f"Physics: Slope={slope}, Manning={manning}")
    print(f"Grid: Nx={Nx}, dt={dt:.4f}s, Nt={Nt}")
    
    # State variables
    h = np.ones(Nx) * h0
    Q = np.ones(Nx) * Q0
    
    # Arrays to store history
    save_interval = 10.0 # Save every 10 seconds for long simulation
    n_save = int(T_total_sim / save_interval) + 1
    
    h_history = np.zeros((n_save, Nx))
    u_history = np.zeros((n_save, Nx))
    t_history = np.linspace(0, T_total_sim, n_save)
    x_history = np.linspace(0, L, Nx)
    
    h_history[0, :] = h
    u_history[0, :] = Q / h
    
    current_time = 0.0
    save_idx = 1
    
    # Helper for Source Term calculation
    def get_source_term(h_curr, Q_curr, local_manning, local_slope):
        h_safe = np.maximum(h_curr, 1e-3)
        u_curr = Q_curr / h_safe
        
        # Manning's friction slope Sf = (n^2 * u * |u|) / R^(4/3)
        # Assuming R approx h for wide channels
        Sf = (local_manning**2 * u_curr * np.abs(u_curr)) / (h_safe**(4/3))
        
        # Source term for momentum equation: g * (S0 - Sf)
        # Note: in MacCormack, source term is added to dQ/dt or d(hu)/dt
        # Momentum eq is d(hu)/dt + d(hu^2 + 0.5gh^2)/dx = g h (S0 - Sf)
        # So when working with Q=hu, the source term is g*h*(S0-Sf) in dQ/dt
        # Here, source term is g*(S0-Sf) applied to du/dt, so we need to multiply by h later, or adjust.
        # Let's use the form for dU/dt = ... + g(S0-Sf) and apply to Q.
        # Source term per unit mass = g * (S0 - Sf)
        # So source term for dQ/dt = g * h * (S0 - Sf)
        # For simplicity, let's keep it as is, but be aware of how Source is applied.
        return g * h_safe * (local_slope - Sf)

    # Solver Loop (MacCormack with Source Term)
    for n in range(Nt):
        if current_time >= T_total_sim:
            break
            
        # Boundary Conditions (Time dependent)
        t_rel = current_time - T_warmup
        
        if current_time < T_warmup:
            Q_in = Q0
        elif t_rel <= 3600.0:
            slope_ramp = 0.5
            Q_in = Q0 * (1.0 - slope_ramp * (t_rel / 3600.0))
        else:
            Q_in = 0.5 * Q0
            
        h_out = h0
        
        # --- Predictor Step ---
        h_p = np.copy(h)
        Q_p = np.copy(Q)
        
        F1 = Q
        F2 = (Q**2 / h) + 0.5 * g * h**2
        
        # Source term at step n
        S_n = get_source_term(h, Q, manning, slope)
        
        # Predictor update includes source term
        h_p[0:-1] = h[0:-1] - (dt/dx) * (F1[1:] - F1[0:-1])
        Q_p[0:-1] = Q[0:-1] - (dt/dx) * (F2[1:] - F2[0:-1]) + dt * S_n[0:-1]
        
        # --- Corrector Step ---
        h_p = np.maximum(h_p, 1e-3) # Ensure h remains positive
        
        F1_p = Q_p
        F2_p = (Q_p**2 / h_p) + 0.5 * g * h_p**2
        
        # Source term at predictor step
        S_p = get_source_term(h_p, Q_p, manning, slope)
        
        h_new = np.zeros_like(h)
        Q_new = np.zeros_like(Q)
        
        # Corrector update includes source term
        h_new[1:] = 0.5 * (h[1:] + h_p[1:] - (dt/dx) * (F1_p[1:] - F1_p[0:-1]))
        Q_new[1:] = 0.5 * (Q[1:] + Q_p[1:] - (dt/dx) * (F2_p[1:] - F2_p[0:-1]) + dt * S_p[1:])
        
        # --- Update BCs ---
        # Inlet (x=0)
        Q_new[0] = Q_in
        
        u_1 = Q_new[1] / h_new[1]
        c_1 = np.sqrt(g * h_new[1])
        R_m = u_1 - 2*c_1
        
        h_guess = h_new[1]
        for _ in range(5):
            f_val = (Q_in/h_guess) - 2*np.sqrt(g*h_guess) - R_m
            df_val = -Q_in/(h_guess**2) - np.sqrt(g)/np.sqrt(h_guess)
            h_guess = h_guess - f_val/df_val
        h_new[0] = h_guess
        
        # Outlet (x=L)
        h_new[-1] = h_out
        
        u_N = Q_new[-2]/h_new[-2]
        c_N = np.sqrt(g*h_new[-2])
        R_plus = u_N + 2*c_N
        u_out = R_plus - 2*np.sqrt(g*h_out)
        Q_new[-1] = u_out * h_out
        
        # Update
        h = h_new
        Q = Q_new
        current_time += dt
        
        # Save frame
        if save_idx < n_save and current_time >= t_history[save_idx]:
            h_history[save_idx, :] = h
            u_history[save_idx, :] = Q / h
            save_idx += 1
            
    print("Solver finished.")
    
    # Save raw data
    os.makedirs("data_raw", exist_ok=True)
    np.save("data_raw/h_history.npy", h_history)
    np.save("data_raw/u_history.npy", u_history)
    np.save("data_raw/x_grid.npy", x_history)
    np.save("data_raw/t_grid.npy", t_history)
    # Explicitly save config dictionary
    np.save("data_raw/config.npy", {"L": L, "T_warmup": T_warmup, "T_sim": T_sim_pinn, "T_total_sim": T_total_sim, "S0": slope, "n_manning": manning})
    
    print("Data saved to data_raw/")
    
    # Plot full history with Variance analysis
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[15, 3, 1], wspace=0.05, hspace=0.3)
    
    # Helper to calculate windowed variance
    def calc_variance_trace(data, t_arr, window_sec=100.0):
        # Determine indices per window
        dt_avg = (t_arr[-1] - t_arr[0]) / (len(t_arr) - 1)
        window_steps = int(window_sec / dt_avg)
        if window_steps < 1: window_steps = 1
        
        vars = []
        ts = []
        
        # Use non-overlapping windows for "clusters" or sliding for smooth
        # User mentioned "clusters" if calc is heavy, but numpy is fast.
        # Let's do sliding window with stride = window_steps (non-overlapping) for clarity
        # to represent "variance within this 100s block".
        
        for i in range(0, len(t_arr) - window_steps, window_steps):
            chunk = data[i : i + window_steps, :]
            # Calculate temporal variance for each spatial point (axis=0 is time)
            # Then take the mean across all spatial points (axis=1 is space)
            # This gives the "average temporal instability" of the field at this time.
            var_val = np.mean(np.var(chunk, axis=0))
            vars.append(var_val)
            # Time point is the center or start of the chunk
            ts.append(t_arr[i]) # Start time
            
        return np.array(vars), np.array(ts)

    # --- 1. Water Depth h ---
    ax_h = fig.add_subplot(gs[0, 0])
    ax_h_var = fig.add_subplot(gs[0, 1], sharey=ax_h)
    ax_h_cb = fig.add_subplot(gs[0, 2])
    
    # Heatmap
    im_h = ax_h.imshow(h_history, aspect='auto', extent=[0, L, T_total_sim, 0], cmap='viridis')
    ax_h.axhline(y=T_warmup, color='k', linestyle='--', label='Warmup End')
    ax_h.set_ylabel('t (s)')
    ax_h.set_xlabel('x (m)')
    ax_h.set_title('Water Depth (Full Simulation)')
    ax_h.legend(loc='upper right')
    
    # Colorbar
    plt.colorbar(im_h, cax=ax_h_cb, label='Water Depth h (m)')
    
    # Variance
    h_vars, h_ts = calc_variance_trace(h_history, t_history, window_sec=100.0)
    # Plot as bar-like fill or line. Line is clearer.
    ax_h_var.plot(h_vars, h_ts, 'k-', linewidth=1)
    ax_h_var.fill_betweenx(h_ts, 0, h_vars, color='gray', alpha=0.5)
    ax_h_var.axhline(y=T_warmup, color='k', linestyle='--')
    ax_h_var.set_xlabel('Variance (100s)')
    ax_h_var.set_xscale('log') # Log scale for variance
    
    # Limit x-axis dynamic range to 1000x
    h_v_max = np.max(h_vars)
    h_v_min = h_v_max / 1000.0
    ax_h_var.set_xlim(h_v_min, h_v_max * 1.2)
    
    # Hide Y labels for var plot as it shares with heatmap
    plt.setp(ax_h_var.get_yticklabels(), visible=False)
    ax_h_var.invert_yaxis() # Match imshow direction
    ax_h_var.grid(True, axis='x', alpha=0.3)

    # --- 2. Discharge Q ---
    ax_q = fig.add_subplot(gs[1, 0])
    ax_q_var = fig.add_subplot(gs[1, 1], sharey=ax_q)
    ax_q_cb = fig.add_subplot(gs[1, 2])
    
    # Calculate Q
    Q_history = h_history * u_history
    
    # Heatmap
    im_q = ax_q.imshow(Q_history, aspect='auto', extent=[0, L, T_total_sim, 0], cmap='viridis')
    ax_q.axhline(y=T_warmup, color='k', linestyle='--', label='Warmup End')
    ax_q.set_ylabel('t (s)')
    ax_q.set_xlabel('x (m)')
    ax_q.set_title('Discharge (Full Simulation)')
    ax_q.legend(loc='upper right')
    
    # Colorbar
    plt.colorbar(im_q, cax=ax_q_cb, label='Discharge Q (m³/s)')
    
    # Variance
    q_vars, q_ts = calc_variance_trace(Q_history, t_history, window_sec=100.0)
    ax_q_var.plot(q_vars, q_ts, 'k-', linewidth=1)
    ax_q_var.fill_betweenx(q_ts, 0, q_vars, color='gray', alpha=0.5)
    ax_q_var.axhline(y=T_warmup, color='k', linestyle='--')
    ax_q_var.set_xlabel('Variance (100s)')
    ax_q_var.set_xscale('log') # Log scale for variance
    
    # Limit x-axis dynamic range to 1000x
    q_v_max = np.max(q_vars)
    q_v_min = q_v_max / 1000.0
    ax_q_var.set_xlim(q_v_min, q_v_max * 1.2)
    
    plt.setp(ax_q_var.get_yticklabels(), visible=False)
    ax_q_var.invert_yaxis()
    ax_q_var.grid(True, axis='x', alpha=0.3)
    
    plt.savefig("data_raw/reference_solution_full.png", bbox_inches='tight')
    print("Reference plot saved to data_raw/reference_solution_full.png")

if __name__ == "__main__":
    generate_saint_venant_data()
