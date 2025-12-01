import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys
import json
import argparse

# Standalone script to avoid TensorFlow conflict

def plot_noisy_distribution_standalone(mode):
    # Determine paths based on execution location
    cwd = os.getcwd()
    if cwd.endswith("src"):
        path_raw = "../data_raw"
        path_config = "../config/best_models/HMC_sv_1d.json"
        path_npy_config = "../data_raw/config.npy"
    else:
        path_raw = "data_raw"
        path_config = "config/best_models/HMC_sv_1d.json"
        path_npy_config = "data_raw/config.npy"

    if not os.path.exists(path_raw):
        print(f"Data path not found: {path_raw}")
        return

    # Load Data
    try:
        h_history = np.load(os.path.join(path_raw, "h_history.npy"))
        u_history = np.load(os.path.join(path_raw, "u_history.npy"))
        t_grid = np.load(os.path.join(path_raw, "t_grid.npy"))
        x_grid = np.load(os.path.join(path_raw, "x_grid.npy"))
        data_config = np.load(path_npy_config, allow_pickle=True).item()
        
        with open(path_config, 'r') as f:
            json_config = json.load(f)
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Parameters
    L = data_config.get("L", 10000.0)
    T_total_sim = data_config.get("T_total_sim", 18000.0)
    T_warmup = data_config.get("T_warmup", 3600.0)
    T_pinn = json_config["physics"]["time"] # 14400
    
    # Noise Parameters
    uncertainty = json_config["uncertainty"]
    noise_h_std = uncertainty.get("noise_h_std_phys", 0.0)
    noise_u_std = uncertainty.get("noise_Q_std_phys", 0.0) 
    
    print(f"Configured Noise: sigma_h={noise_h_std}, sigma_u={noise_u_std}")

    # Create Meshgrid for Masking
    Tv, Xv = np.meshgrid(t_grid, x_grid, indexing='ij')
    
    # Normalize coordinates for masking logic
    X_norm = Xv / L
    T_norm = (Tv - T_warmup) / T_pinn
    
    # Mask Logic
    mask_spatial = (X_norm > 0.1) & (X_norm < 0.9)
    mask_temporal = (T_norm > 0.1) & (T_norm < 0.9)
    mask_training = mask_spatial & mask_temporal 

    # Generate Base Noise
    np.random.seed(42)
    noise_h_base = np.random.normal(0, noise_h_std, h_history.shape)
    noise_u_base = np.random.normal(0, noise_u_std, u_history.shape)

    apply_mask = (mode == 'masked')
    
    if apply_mask:
        h_noisy = h_history + noise_h_base * mask_training
        u_noisy = u_history + noise_u_base * mask_training
        filename_suffix = "masked"
    else:
        h_noisy = h_history + noise_h_base
        u_noisy = u_history + noise_u_base
        filename_suffix = "full_domain"

    print(f"Generating {filename_suffix} plot...")

    # --- Plotting ---
    plt.rcParams.update({'font.size': 10})
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[15, 3, 1], wspace=0.05, hspace=0.3)
    
    # Helper for variance
    def calc_variance_trace(data, t_arr, window_sec=100.0):
        dt_avg = (t_arr[-1] - t_arr[0]) / (len(t_arr) - 1)
        window_steps = int(window_sec / dt_avg)
        if window_steps < 1: window_steps = 1
        vars_list = []
        ts = []
        for i in range(0, len(t_arr) - window_steps, window_steps):
            chunk = data[i : i + window_steps, :]
            vars_list.append(np.mean(np.var(chunk, axis=0)))
            ts.append(t_arr[i])
        return np.array(vars_list), np.array(ts)

    # Draw Mask Box Helper
    def draw_mask_box(ax):
        if apply_mask:
            t_start = T_warmup + 0.1 * T_pinn
            t_end = T_warmup + 0.9 * T_pinn
            # Draw rectangle
            rect = plt.Rectangle((0.1*L, t_start), 0.8*L, t_end-t_start, 
                                linewidth=1.5, edgecolor='white', facecolor='none', linestyle='--')
            ax.add_patch(rect)

    # ==========================================
    # 1. Water Depth h
    # ==========================================
    ax_h = fig.add_subplot(gs[0, 0])
    ax_h_var = fig.add_subplot(gs[0, 1], sharey=ax_h)
    ax_h_cb = fig.add_subplot(gs[0, 2])
    
    im_h = ax_h.imshow(h_noisy, aspect='auto', origin='lower', 
                       extent=[0, L, 0, T_total_sim], cmap='viridis')
    
    ax_h.axhline(y=T_warmup, color='k', linestyle='--', linewidth=1.5, label='Warmup End')
    draw_mask_box(ax_h)
    
    ax_h.set_ylabel('t (s)')
    ax_h.set_xlabel('x (m)')
    ax_h.set_title('Water Depth (Full Simulation)')
    ax_h.legend(loc='upper right', fancybox=True, framealpha=0.8)
    
    # Custom noise text box
    if apply_mask:
        noise_text = r"Noise $\sigma_h$=" + f"{noise_h_std} m\n(Applied in Dashed Box)"
    else:
        noise_text = r"Noise $\sigma_h$=" + f"{noise_h_std} m\n(Applied Everywhere)"
        
    ax_h.text(0.02, 0.95, noise_text, transform=ax_h.transAxes, 
              color='white', fontsize=11, verticalalignment='top',
              bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=4))

    plt.colorbar(im_h, cax=ax_h_cb, label='Water Depth h (m)')
    
    # Variance
    h_vars, h_ts = calc_variance_trace(h_noisy, t_grid)
    ax_h_var.plot(h_vars, h_ts, 'k-', linewidth=1)
    ax_h_var.fill_betweenx(h_ts, 0, h_vars, color='silver', alpha=1.0)
    ax_h_var.axhline(y=T_warmup, color='k', linestyle='--', linewidth=1.5)
    ax_h_var.axvline(x=noise_h_std**2, color='k', linestyle=':', linewidth=1.0)

    ax_h_var.set_xlabel('Variance (100s)')
    ax_h_var.set_xscale('log')
    ax_h_var.set_ylim(0, T_total_sim) 
    plt.setp(ax_h_var.get_yticklabels(), visible=False)
    ax_h_var.grid(True, axis='x', alpha=0.3, which='both')

    # ==========================================
    # 2. Velocity u
    # ==========================================
    ax_u = fig.add_subplot(gs[1, 0])
    ax_u_var = fig.add_subplot(gs[1, 1], sharey=ax_u)
    ax_u_cb = fig.add_subplot(gs[1, 2])
    
    im_u = ax_u.imshow(u_noisy, aspect='auto', origin='lower',
                       extent=[0, L, 0, T_total_sim], cmap='viridis') 
    
    ax_u.axhline(y=T_warmup, color='k', linestyle='--', linewidth=1.5, label='Warmup End')
    draw_mask_box(ax_u)

    ax_u.set_ylabel('t (s)')
    ax_u.set_xlabel('x (m)')
    ax_u.set_title('Velocity (Full Simulation)')
    ax_u.legend(loc='upper right', fancybox=True, framealpha=0.8)

    if apply_mask:
        noise_text_u = r"Noise $\sigma_u$=" + f"{noise_u_std} m/s\n(Applied in Dashed Box)"
    else:
        noise_text_u = r"Noise $\sigma_u$=" + f"{noise_u_std} m/s\n(Applied Everywhere)"

    ax_u.text(0.02, 0.95, noise_text_u, transform=ax_u.transAxes, 
              color='white', fontsize=11, verticalalignment='top',
              bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=4))

    plt.colorbar(im_u, cax=ax_u_cb, label='Velocity u (m/s)')
    
    # Variance
    u_vars, u_ts = calc_variance_trace(u_noisy, t_grid)
    ax_u_var.plot(u_vars, u_ts, 'k-', linewidth=1)
    ax_u_var.fill_betweenx(u_ts, 0, u_vars, color='silver', alpha=1.0)
    ax_u_var.axhline(y=T_warmup, color='k', linestyle='--', linewidth=1.5)
    ax_u_var.axvline(x=noise_u_std**2, color='k', linestyle=':', linewidth=1.0)

    ax_u_var.set_xlabel('Variance (100s)')
    ax_u_var.set_xscale('log')
    ax_u_var.set_ylim(0, T_total_sim)
    plt.setp(ax_u_var.get_yticklabels(), visible=False)
    ax_u_var.grid(True, axis='x', alpha=0.3, which='both')
    
    save_path = os.path.join(path_raw, f"noisy_solution_{filename_suffix}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['masked', 'full'], required=True)
    args = parser.parse_args()
    plot_noisy_distribution_standalone(args.mode)