import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import tensorflow as tf

# Add src to path
sys.path.append(os.path.dirname(__file__))
from utility import set_directory, load_json, switch_dataset, switch_equation
from setup import Parser, Param, Dataset
from networks import BayesNN
from networks.Theta import Theta # Moved to top


def compute_fdm_residual(X, T_grid, H, U, g=9.81, S0=0.001, n_manning=0.03):
    """
    Compute PDE residuals using Finite Difference Method.
    """
    # Gradients (Central Difference)
    dt = np.gradient(T_grid, axis=0)
    dx = np.gradient(X, axis=1)
    dt[dt==0] = 1e-8
    dx[dx==0] = 1e-8
    
    h_t = np.gradient(H, axis=0) / dt
    h_x = np.gradient(H, axis=1) / dx
    u_t = np.gradient(U, axis=0) / dt
    u_x = np.gradient(U, axis=1) / dx
    
    # Source Terms
    h_safe = np.maximum(H, 0.1)
    Sf = (n_manning**2 * U * np.abs(U)) / (h_safe**(4/3))
    
    # Residuals
    Res_C = h_t + h_x * U + H * u_x
    Res_M = u_t + U * u_x + g * h_x + g * (Sf - S0)
    
    # Scales (Max Terms)
    # Continuity terms
    tc = [np.abs(h_t), np.abs(h_x * U), np.abs(H * u_x)]
    scale_c = np.max(tc, axis=0)
    
    # Momentum terms
    tm = [np.abs(u_t), np.abs(U * u_x), np.abs(g * h_x), np.abs(g * (Sf - S0))]
    scale_m = np.max(tm, axis=0)
    
    return Res_C, Res_M, scale_c, scale_m

def compare_residuals():
    set_directory()
    
    # 1. Find latest HMC run & Load PINN Data
    parent_path = "../outs/SaintVenant1D/SaintVenant1D"
    if not os.path.exists(parent_path):
        print("Path not found.")
        return

    folders = [os.path.join(parent_path, f) for f in os.listdir(parent_path) if f.startswith("HMC")]
    folders = [f for f in folders if os.path.isdir(f) and not f.endswith("/HMC") and not f.endswith(os.sep + "HMC")]
    
    if not folders:
        print("No HMC output found.")
        return
        
    path_folder = sorted(folders, key=os.path.getmtime)[-1]
    print(f"Comparing results from: {path_folder}")
    
    # Load Config (to initialize Param and BayesNN)
    config_file = "best_models/HMC_sv_1d_short" # Assuming this was used
    config_data = load_json(config_file)
    
    # Create dummy args for Param
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
    
    params = Param(config_data, args)
    data_config = switch_dataset(params.problem, params.case_name)
    params.data_config = data_config
    
    # Instantiate BayesNN (to compute Loss)
    equation = switch_equation(params.problem)
    bayes_nn = BayesNN(params, equation)
    
    # Load PINN trained weights
    path_thetas = os.path.join(path_folder, "thetas")
    # There should be only one theta saved by the current HMC setup (the mean or last sample)
    # Assuming thetas are in theta_001/w_001.npy etc.
    pinn_thetas = []
    
    # Find the last Theta folder (if multiple)
    theta_subfolders = [f for f in os.listdir(path_thetas) if f.startswith("theta_")]
    if theta_subfolders:
        latest_theta_folder = sorted(theta_subfolders)[-1] # theta_080 for 80 samples or similar
        
        target_folder = os.path.join(path_thetas, latest_theta_folder)
        if os.path.isdir(target_folder):
            # Load weights/biases manually
            def load_list(p, name):
                l = []
                files = sorted([f for f in os.listdir(p) if f.startswith(name + "_")])
                for f in files:
                    l.append(np.load(os.path.join(p, f)))
                return l

            ws = load_list(target_folder, "w")
            bs = load_list(target_folder, "b")
            
            theta_vals = []
            for w, b in zip(ws, bs):
                theta_vals.append(tf.convert_to_tensor(w, dtype=tf.float32))
                theta_vals.append(tf.convert_to_tensor(b, dtype=tf.float32))
            pinn_thetas = [Theta(theta_vals)] # Wrap in a list for nn_params setter
    
    if pinn_thetas:
        bayes_nn.nn_params = pinn_thetas[0] # Set the PINN's parameters
    else:
        print("Warning: PINN parameters not loaded for Loss calculation.")
    
    # 2. Load Data for FDM & Ground Truth Loss
    # Prediction
    sol_NN = np.load(os.path.join(path_folder, "values", "sol_NN.npy"))
    
    # Coordinates & True Values
    possible_data_paths = [
        "../data/SaintVenant1D/simple",
        "../data/SaintVenant1D/SaintVenant1D",
        "../data/SaintVenant1D/SaintVenant1D_simple"
    ]
    
    data_dir = None
    for p in possible_data_paths:
        if os.path.exists(os.path.join(p, "dom_test.npy")):
            data_dir = p
            break
            
    if data_dir is None:
        print("Data directory not found.")
        return
        
    print(f"Loading ground truth from: {data_dir}")
    dom_test = np.load(os.path.join(data_dir, "dom_test.npy"))
    sol_test = np.load(os.path.join(data_dir, "sol_test.npy"), allow_pickle=True)
    
    print(f"DEBUG: sol_test type: {type(sol_test)}")
    print(f"DEBUG: sol_test shape: {getattr(sol_test, 'shape', 'N/A')}")
    
    # If it's 0-d array containing an object (rare but possible with pickle)
    if sol_test.ndim == 0:
        print("DEBUG: sol_test is 0-d array, extracting...")
        sol_test = sol_test.item()
        print(f"DEBUG: extracted type: {type(sol_test)}")
    
    # 3. Reconstruct Grid (Same for FDM and PINN Loss)
    L = 10000.0
    T = 14400.0
    
    # Denormalize coordinates
    x_flat = dom_test[:, 0] * L
    t_flat = dom_test[:, 1] * T
    
    # Sort
    sort_idx = np.lexsort((x_flat, t_flat))
    x_sorted = x_flat[sort_idx]
    t_sorted = t_flat[sort_idx]
    
    # Grid Size
    unique_t = np.unique(t_sorted)
    unique_x = np.unique(x_sorted)
    Nt = len(unique_t)
    Nx = len(unique_x)
    
    X = x_sorted.reshape(Nt, Nx)
    T_grid = t_sorted.reshape(Nt, Nx)
    
    # Prepare Fields for FDM
    # Prediction
    H_pred = sol_NN[:, 0][sort_idx].reshape(Nt, Nx)
    U_pred = sol_NN[:, 1][sort_idx].reshape(Nt, Nx)
    
    # Exact
    H_true = sol_test[:, 0][sort_idx].reshape(Nt, Nx)
    U_true = sol_test[:, 1][sort_idx].reshape(Nt, Nx)
    
    # 4. Compute FDM Residuals
    print("Computing FDM Residuals for Prediction...")
    Rc_p_fdm, Rm_p_fdm, Sc_p_fdm, Sm_p_fdm = compute_fdm_residual(X, T_grid, H_pred, U_pred)
    
    print("Computing FDM Residuals for Ground Truth...")
    Rc_t_fdm, Rm_t_fdm, Sc_t_fdm, Sm_t_fdm = compute_fdm_residual(X, T_grid, H_true, U_true)
    
    # 5. Comparison Statistics (Inner Points)
    mask = np.ones_like(Rc_p_fdm, dtype=bool)
    mask[0,:] = False; mask[-1,:] = False
    mask[:,0] = False; mask[:,-1] = False
    
    def get_fdm_stats(res, scale):
        r = res[mask]
        s = scale[mask]
        abs_mean = np.mean(np.abs(r))
        s_safe = s.copy()
        s_safe[s_safe < 1e-8] = 1e-8
        rel_mean = np.mean(np.abs(r) / s_safe) * 100
        return abs_mean, rel_mean

    abs_cp_fdm, rel_cp_fdm = get_fdm_stats(Rc_p_fdm, Sc_p_fdm)
    abs_mp_fdm, rel_mp_fdm = get_fdm_stats(Rm_p_fdm, Sm_p_fdm)
    abs_ct_fdm, rel_ct_fdm = get_fdm_stats(Rc_t_fdm, Sc_t_fdm)
    abs_mt_fdm, rel_mt_fdm = get_fdm_stats(Rm_t_fdm, Sm_t_fdm)


    # 5. Compute PINN Loss (AD Residuals)
    # To compute PINN Loss, we need a Dataset object
    # For Ground Truth, we need a Dataset with 0 noise
    # For PINN Prediction, we need a Dataset with the training noise configuration
    
    # --- Ground Truth PINN Loss ---
    # Create a parameter object with 0 noise for Ground Truth Dataset
    gt_params_config = config_data.copy()
    gt_params_config["uncertainty"]["noise_h_std_phys"] = 0.0
    gt_params_config["uncertainty"]["noise_Q_std_phys"] = 0.0
    gt_params_config["utils"]["gen_flag"] = False # Do not regenerate data, use existing

    gt_params = Param(gt_params_config, args)
    gt_data_config = switch_dataset(gt_params.problem, gt_params.case_name)
    gt_params.data_config = gt_data_config
    
    # This dataset will have no noise because add_noise=False
    # It will use data_config.values["u"] which is the true solution
    gt_dataset = Dataset(gt_params, add_noise=False)
    gt_dataset.normalize_dataset() # Normalize to match model expectation
    
    # Set model's normalization coefficients to match this GT dataset
    bayes_nn.norm_coeff = gt_dataset.norm_coeff 
    
    # Calculate Ground Truth Loss (model's view of perfect solution)
    # The current nn_params are for the PINN prediction. This loss is "how well does the PINN-learned network fit the perfect data?"
    gt_loss_pst, gt_loss_llk = bayes_nn.metric_total(gt_dataset)

    # --- PINN Prediction Loss (from training log) ---
    # We load the final loss from the training log
    path_log = os.path.join(path_folder, "log")
    keys = []
    with open(os.path.join(path_log, "keys.txt"), 'r') as f:
        for line in f:
            keys.append(line.strip())
    
    posterior_log = np.load(os.path.join(path_log, "posterior.npy"))
    pinn_loss_pst = {}
    for i, key in enumerate(keys):
        pinn_loss_pst[key] = posterior_log[i, -1] # Last value

    print("\n" + "="*80)
    print(f"{ 'Metric':<20} | { 'PINN Predicted (FDM)':<25} | { 'Ground Truth (FDM)':<25} | { 'Ratio (Pred/True)':<10}")
    print("-" * 80)
    print(f"{ 'Cont Abs Res':<20} | {abs_cp_fdm:.4e}                    | {abs_ct_fdm:.4e}                    | {abs_cp_fdm/abs_ct_fdm:.2f}x")
    print(f"{ 'Mom  Abs Res':<20} | {abs_mp_fdm:.4e}                    | {abs_mt_fdm:.4e}                    | {abs_mp_fdm/abs_mt_fdm:.2f}x")
    print("-" * 80)
    print(f"{ 'Cont Rel Err':<20} | {rel_cp_fdm:.2f}%                     | {rel_ct_fdm:.2f}%                     | {rel_cp_fdm/rel_ct_fdm:.2f}x")
    print(f"{ 'Mom  Rel Err':<20} | {rel_mp_fdm:.2f}%                     | {rel_mt_fdm:.2f}%                     | {rel_mp_fdm/rel_mt_fdm:.2f}x")
    print("="*80)

    print("\n" + "="*80)
    print(f"{ 'PINN Loss (AD) MSE':<20} | { 'PINN Predicted (Final)':<25} | { 'Ground Truth (Ideal)':<25} | { 'Ratio (Pred/Ideal)':<10}")
    print("-" * 80)
    # Adjusting for keys that might not exist for ideal GT loss
    def print_loss_row(key):
        pinn_val = pinn_loss_pst.get(key, 0.0)
        gt_val = gt_loss_pst[0].get(key, 0.0) # gt_loss_pst is (pst_dict, llk_dict)
        ratio = pinn_val / (gt_val + 1e-10) if gt_val != 0 else np.nan
        print(f"{ key:<20} | {pinn_val:.4e}                    | {gt_val:.4e}                    | {ratio:.2f}x")

    print_loss_row('data_u')
    print_loss_row('data_b')
    print_loss_row('pde')
    print_loss_row('prior')
    print_loss_row('Total')

    print("="*80)

    # Plot Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plotting helpers
    def plot_map(ax, data, title, cmap='viridis'):
        im = ax.pcolormesh(X, T_grid, data, cmap=cmap, shading='auto')
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
    
    # Difference Maps
    diff_h = H_pred - H_true
    plot_map(axes[0,0], diff_h, "Error H (Pred - True)", cmap='RdBu_r')
    
    diff_mom_res = np.abs(Rm_p_fdm) - np.abs(Rm_t_fdm)
    plot_map(axes[0,1], diff_mom_res, "Diff Momentum Res (|Rp| - |Rt|)", cmap='RdBu_r')
    
    # Residual Maps
    vmax = np.percentile(np.abs(Rm_p_fdm), 95)
    plot_map(axes[1,0], Rm_p_fdm, "Momentum Res (Pred)", cmap='RdBu_r')
    plot_map(axes[1,1], Rm_t_fdm, "Momentum Res (True)", cmap='RdBu_r')
    
    plt.tight_layout()
    save_path = os.path.join(path_folder, "residual_comparison.png")
    plt.savefig(save_path)
    print(f"\nComparison plot saved to: {save_path}")

if __name__ == "__main__":
    compare_residuals()