import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from setup import Param, Dataset
from networks.BayesNN import BayesNN
from networks.Theta import Theta
from utility import load_json, switch_dataset, switch_equation, set_directory

def diagnose_error_artifact():
    set_directory()
    
    # Configuration
    folder_name = "HMC_2025.11.26-17.06.48"
    config_file = "best_models/HMC_sv_1d"
    
    # Setup Model (Simplified)
    print("Setting up model...")
    config_path = os.path.join(os.path.dirname(__file__), "../config/best_models/HMC_sv_1d.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    class Args:
        def __init__(self):
            self.config = config_file
            self.problem = None
            self.case_name = None
            self.method = None
            self.epochs = None
            self.save_flag = True
            self.gen_flag = False 
            self.debug_flag = False
            self.random_seed = 42
    args = Args()
    params = Param(config_dict, args)
    
    data_config = switch_dataset(params.problem, params.case_name)
    params.data_config = data_config
    dataset = Dataset(params)
    
    equation = switch_equation(params.problem)
    bayes_nn = BayesNN(params, equation)
    
    bayes_nn.u_coeff = dataset.norm_coeff["sol_mean"], dataset.norm_coeff["sol_std"]
    bayes_nn.f_coeff = dataset.norm_coeff["par_mean"], dataset.norm_coeff["par_std"]
    bayes_nn.norm_coeff = dataset.norm_coeff
    
    # Load Weights
    ckpt_path = f"../outs/SaintVenant1D/SaintVenant1D/{folder_name}/checkpoints/checkpoint_latest.npy"
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found, falling back to pretrained.")
        ckpt_path = "../pretrained_models/pretrained_SaintVenant1D_simple_ADAM.npy"
    
    print(f"Loading weights from {ckpt_path}...")
    loaded_values = np.load(ckpt_path, allow_pickle=True)
    theta_values = [tf.convert_to_tensor(v, dtype=tf.float32) for v in loaded_values]
    bayes_nn.nn_params = Theta(theta_values)
    bayes_nn.thetas = [bayes_nn.nn_params]

    # --- Diagnostics ---
    print("Diagnosing...")
    L = params.physics["length"]
    T = params.physics["time"] # 14400
    Nx = 200
    Nt = 200
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    X, T_mesh = np.meshgrid(x, t)
    
    x_flat = X.flatten()
    t_flat = T_mesh.flatten()
    x_norm = x_flat / L
    t_norm = t_flat / T
    
    inputs = np.stack([x_norm, t_norm], axis=1)
    
    # Exact & Pred
    exact_res = data_config.values["u"]([x_flat, t_norm])
    h_ex = exact_res[0].flatten()
    
    preds = bayes_nn.mean_and_std(inputs)
    h_nn = preds["sol_NN"][:, 0]
    
    err_h = h_nn - h_ex
    err_grid = err_h.reshape(Nt, Nx)
    
    # Plotting
    save_path = f"../outs/SaintVenant1D/SaintVenant1D/{folder_name}/plot/diagnosis.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Error Sign Map
    # Red = Positive Error (NN > Ex), Blue = Negative Error (NN < Ex)
    ax1 = axes[0]
    p1 = ax1.pcolor(X, T_mesh, err_grid, cmap='RdBu_r', vmin=-np.max(np.abs(err_grid)), vmax=np.max(np.abs(err_grid)), shading='auto')
    fig.colorbar(p1, ax=ax1, label='Error (h_nn - h_ex)')
    ax1.set_title('Signed Error Map')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('t (s)')
    # ax1.invert_yaxis() # Keep 0 at bottom
    
    # 2. Slice at x = L/2
    ax2 = axes[1]
    mid_idx = Nx // 2
    t_slice = t
    h_nn_slice = h_nn.reshape(Nt, Nx)[:, mid_idx]
    h_ex_slice = h_ex.reshape(Nt, Nx)[:, mid_idx]
    err_slice = h_nn_slice - h_ex_slice
    
    ax2.plot(t_slice, h_ex_slice, 'k-', label='Exact')
    ax2.plot(t_slice, h_nn_slice, 'b--', label='NN')
    ax2.plot(t_slice, err_slice, 'r:', label='Error')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_title(f'Time Slice at x = {x[mid_idx]:.0f} m')
    ax2.set_xlabel('t (s)')
    ax2.set_ylabel('h (m)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Diagnosis saved to {save_path}")
    
    # Print Timing Info
    print("\n--- Timing Verification ---")
    print(f"PINN Time Domain: [0, {T}] s")
    print("Exact Solution interpolation uses t_norm * T.")
    print(f"Config T_warmup: {3600.0} (Assumption)")
    print("If artifact is at t ~ 3200s:")
    print(f"  It corresponds to {3200}s in PINN domain.")
    print(f"  It corresponds to {3200 + 3600} = 6800s in Physical Simulation.")
    
    # Check if error crosses zero
    zero_crossings = np.where(np.diff(np.sign(err_slice)))
    if len(zero_crossings[0]) > 0:
        cross_t = t_slice[zero_crossings[0]]
        print(f"Zero crossings detected at t = {cross_t}")
    else:
        print("No zero crossings detected in the slice.")

if __name__ == "__main__":
    diagnose_error_artifact()
