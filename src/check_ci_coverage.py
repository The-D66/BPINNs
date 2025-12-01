import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(__file__))
from utility import set_directory, load_json, switch_dataset, switch_equation
from setup import Param, Dataset
from networks import BayesNN
from networks.Theta import Theta

def check_coverage():
    # Setup
    set_directory()
    
    # We need to find the latest output folder for HMC to load samples
    # Assuming standard path structure: ../outs/SaintVenant1D/SaintVenant1D/HMC_...
    base_path = "../outs/SaintVenant1D/SaintVenant1D"
    if not os.path.exists(base_path):
        print("No output directory found.")
        return

    # Manually select High Quality Runs (FFE + Viscosity + Balanced Weights)
    target_folders_names = [
        'HMC_2025.12.01-12.38.01' 
    ]
    
    base_path = "../outs/SaintVenant1D/internal_pde"
    target_folders = [os.path.join(base_path, f) for f in target_folders_names]
    print(f"Loading samples from {len(target_folders)} chains: {target_folders_names}")

    # Load Config used for that run
    config_file = "best_models/HMC_sv_1d_internal"
    config = load_json(config_file)

    # Dummy Args
    class Args:
        def __init__(self):
            self.config = config_file
            self.problem = None
            self.case_name = None
            self.method = None
            self.epochs = None
            self.save_flag = False
            self.gen_flag = False
            self.debug_flag = False
            self.random_seed = 42
    args = Args()
    
    params = Param(config, args)
    data_config = switch_dataset(params.problem, params.case_name)
    params.data_config = data_config

    # Dataset
    dataset = Dataset(params)
    
    # Model
    equation = switch_equation(params.problem)
    bayes_nn = BayesNN(params, equation)
    
    # Set Norm Coeffs
    bayes_nn.u_coeff = dataset.norm_coeff["sol_mean"], dataset.norm_coeff["sol_std"]
    bayes_nn.f_coeff = dataset.norm_coeff["par_mean"], dataset.norm_coeff["par_std"]
    bayes_nn.norm_coeff = dataset.norm_coeff

    # Load Samples
    thetas = []
    
    for path_folder_single_run in target_folders: # Renamed for clarity
        path_thetas = os.path.join(path_folder_single_run, "thetas")
        if os.path.exists(path_thetas):
            for folder in sorted(os.listdir(path_thetas)):
                folder_path = os.path.join(path_thetas, folder)
                if not os.path.isdir(folder_path): continue
                
                try:
                    # Helper to load list
                    def load_list(p, name):
                        l = []
                        files = sorted([f for f in os.listdir(p) if f.startswith(name + "_")])
                        for f in files:
                            l.append(np.load(os.path.join(p, f)))
                        return l

                    ws = load_list(folder_path, "w")
                    bs = load_list(folder_path, "b")
                    
                    theta_vals = []
                    for w, b in zip(ws, bs):
                        theta_vals.append(tf.convert_to_tensor(w, dtype=tf.float32))
                        theta_vals.append(tf.convert_to_tensor(b, dtype=tf.float32))
                    thetas.append(Theta(theta_vals))
                except Exception as e:
                    print(f"Error loading {folder}: {e}")

    if not thetas:
        print("No samples loaded.")
        return
        
    print(f"Loaded {len(thetas)} samples.")
    bayes_nn.thetas = thetas # Assign all loaded thetas to the BayesNN model

    # Evaluate on Test Data
    L = params.physics["length"]
    T = params.physics["time"]
    
    target_locs_km = [0, 2, 4, 6, 8, 10]
    num_t = 100
    t_h_max = T / 3600.0
    t_h = np.linspace(0, t_h_max, num_t)
    t_norm = t_h / t_h_max
    
    total_points = 0
    inside_h = 0
    inside_Q = 0
    
    print("--- Coverage Analysis (2 Sigma, ~95%) ---")
    
    # Prepare data for plotting
    plot_data_list_h = []
    plot_data_list_Q = []

    for x_km in target_locs_km:
        x_norm_val = (x_km * 1000.0) / L
        x_col = np.full(num_t, x_norm_val)
        query_points = np.stack([x_col, t_norm], axis=1)
        
        # Predictions (mean and std from all loaded samples)
        preds = bayes_nn.mean_and_std(query_points)
        h_mean = preds["sol_NN"][:, 0]
        u_mean = preds["sol_NN"][:, 1]
        h_std  = preds["sol_std"][:, 0]
        u_std  = preds["sol_std"][:, 1]
        
        # Exact values
        exact_res = data_config.values["u"]([x_col, t_norm])
        h_ex = exact_res[0].flatten()
        u_ex = exact_res[1].flatten()
        Q_ex = h_ex * u_ex
        
        # Derived Q stats
        Q_mean = h_mean * u_mean
        Q_std = np.sqrt( (u_mean * h_std)**2 + (h_mean * u_std)**2 )
        
        # Check Coverage h
        lower_h = h_mean - 2 * h_std
        upper_h = h_mean + 2 * h_std
        is_in_h = (h_ex >= lower_h) & (h_ex <= upper_h)
        
        # Check Coverage Q
        lower_Q = Q_mean - 2 * Q_std
        upper_Q = Q_mean + 2 * Q_std
        is_in_Q = (Q_ex >= lower_Q) & (Q_ex <= upper_Q)
        
        # Stats
        n = len(h_ex)
        total_points += n
        inside_h += np.sum(is_in_h)
        inside_Q += np.sum(is_in_Q)
        
        print(f"x={x_km}km: h_cov={np.mean(is_in_h)*100:.1f}%, Q_cov={np.mean(is_in_Q)*100:.1f}% | Mean Width h={np.mean(upper_h-lower_h):.4f}")

        # Store data for plotting
        plot_data_list_h.append({
            "x_km": x_km, "t_h": t_h, "h_ex": h_ex, "h_mean": h_mean, "h_std": h_std,
            "train_t": [], "train_h": [] # No training data plotted here
        })
        plot_data_list_Q.append({
            "x_km": x_km, "t_h": t_h, "Q_ex": Q_ex, "Q_mean": Q_mean, "Q_std": Q_std,
            "train_t": [], "train_Q": [] # No training data plotted here
        })

    print("-" * 30)
    print(f"Total Coverage h: {inside_h/total_points*100:.1f}%")
    print(f"Total Coverage Q: {inside_Q/total_points*100:.1f}%")
    print(f"Target: ~95%")

    # --- Plotting ---
    # Reuse plotter logic from postprocessing/Plotter.py
    # Since we don't have a Plotter object here, we'll implement a basic version
    
    # Get latest path_folder for saving plots
    latest_run_folder = sorted(target_folders, key=os.path.getmtime)[-1]
    plot_save_dir = os.path.join(latest_run_folder, "plot_coverage_analysis")
    os.makedirs(plot_save_dir, exist_ok=True)

    # Plot Water Depth h
    num_locs = len(plot_data_list_h)
    cols = 2
    rows = (num_locs + 1) // cols
    
    fig_h, axes_h = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=True, sharey=True)
    axes_h = axes_h.flatten()
    
    for i, data in enumerate(plot_data_list_h):
        ax = axes_h[i]
        t = data["t_h"]
        ax.plot(t, data["h_ex"], 'k-', linewidth=1.5, label='Exact')
        ax.plot(t, data["h_mean"], 'b--', linewidth=1.5, label='NN Mean')
        lower = data["h_mean"] - 2 * data["h_std"]
        upper = data["h_mean"] + 2 * data["h_std"]
        ax.fill_between(t, lower, upper, color='blue', alpha=0.2, label='95% CI')
        ax.set_title(f"x = {data['x_km']:.1f} km")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='best', fontsize='small')

    fig_h.supxlabel("Time (h)")
    fig_h.supylabel("Water Depth h (m)")
    for i in range(num_locs, len(axes_h)): axes_h[i].axis('off') # Hide empty subplots
    fig_h.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, "time_series_h_coverage.png"))
    plt.close(fig_h)

    # Plot Discharge Q
    fig_q, axes_q = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=True, sharey=True)
    axes_q = axes_q.flatten()
    
    for i, data in enumerate(plot_data_list_Q):
        ax = axes_q[i]
        t = data["t_h"]
        ax.plot(t, data["Q_ex"], 'k-', linewidth=1.5, label='Exact')
        ax.plot(t, data["Q_mean"], 'b--', linewidth=1.5, label='NN Mean')
        lower = data["Q_mean"] - 2 * data["Q_std"]
        upper = data["Q_mean"] + 2 * data["Q_std"]
        ax.fill_between(t, lower, upper, color='blue', alpha=0.2, label='95% CI')
        ax.set_title(f"x = {data['x_km']:.1f} km")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='best', fontsize='small')

    fig_q.supxlabel("Time (h)")
    fig_q.supylabel("Discharge Q (m³/s)")
    for i in range(num_locs, len(axes_q)): axes_q[i].axis('off') # Hide empty subplots
    fig_q.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, "time_series_Q_coverage.png"))
    plt.close(fig_q)

    print(f"\nCoverage plots saved to: {plot_save_dir}")

if __name__ == "__main__":
    check_coverage()