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
from equations.Operators import Operators

def plot_loss_distribution():
    # Ensure we are in the src directory for relative paths to work
    set_directory()

    # Configuration
    folder_name = "HMC_2025.11.26-17.06.48"
    # base_path is relative to src now
    base_path = f"../outs/SaintVenant1D/SaintVenant1D/{folder_name}" 
    
    # Folder for saving plots
    plot_save_path = "../outs/SaintVenant1D/SaintVenant1D/Loss_Analysis"
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    plot_path = os.path.join(plot_save_path, "plot")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
        
    config_file = "best_models/HMC_sv_1d" 
    
    # Setup Model
    print("Setting up model...")
    print(f"CWD: {os.getcwd()}")
    # Dataset loads from ../data, so if we are in src, it looks in (root)/data. Correct.
    
    # Robust config loading
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
    
    # Load Dataset for normalization stats
    data_config = switch_dataset(params.problem, params.case_name)
    params.data_config = data_config
    dataset = Dataset(params)
    
    # Build Model
    equation = switch_equation(params.problem)
    bayes_nn = BayesNN(params, equation)
    
    # Set norm coeffs
    bayes_nn.u_coeff = dataset.norm_coeff["sol_mean"], dataset.norm_coeff["sol_std"]
    bayes_nn.f_coeff = dataset.norm_coeff["par_mean"], dataset.norm_coeff["par_std"]
    bayes_nn.norm_coeff = dataset.norm_coeff
    
    # Load Weights
    # Assuming base_path is relative to src: ../outs/...
    # Checkpoint is in checkpoints/
    ckpt_path = os.path.join(base_path, "checkpoints", "checkpoint_latest.npy")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        # Fallback to pretrained
        ckpt_path = "../pretrained_models/pretrained_SaintVenant1D_simple_ADAM.npy"
        if not os.path.exists(ckpt_path):
             print("No weights found.")
             return

    print(f"Loading weights from {ckpt_path}...")
    loaded_values = np.load(ckpt_path, allow_pickle=True)
    theta_values = [tf.convert_to_tensor(v, dtype=tf.float32) for v in loaded_values]
    bayes_nn.nn_params = Theta(theta_values)
    # Important: Populate thetas list for mean_and_std to work
    bayes_nn.thetas = [bayes_nn.nn_params]
    
    # Generate Grid
    print("Generating evaluation grid...")
    L = params.physics["length"]
    T = params.physics["time"]
    Nx = 200
    Nt = 200
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    X, T_mesh = np.meshgrid(x, t)
    
    # Flatten for prediction
    x_flat = X.flatten()
    t_flat = T_mesh.flatten()
    
    # Normalize inputs
    x_norm = x_flat / L
    t_norm = t_flat / T 
    
    inputs = np.stack([x_norm, t_norm], axis=1)
    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    
    # Forward Pass & PDE Residual Calculation
    print("Computing residuals...")
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs_tf)
        # Forward
        out_sol, _ = bayes_nn.forward(inputs_tf)
        
        # PDE Residual (re-using model logic)
        # comp_residual returns (N, 2) tensor [cont_res, mom_res]
        res_tf = bayes_nn.pinn.comp_residual(inputs_tf, out_sol, None, tape)
    
    res_np = res_tf.numpy()
    res_cont = res_np[:, 0].reshape(Nt, Nx)
    res_mom = res_np[:, 1].reshape(Nt, Nx)
    
    # Calculate PDE Loss Map (Squared Residual)
    pde_loss_map = res_cont**2 + res_mom**2
    
    # Data Mismatch Calculation
    print("Computing data mismatch...")
    # Get Exact Solution
    exact_res = data_config.values["u"]([x_flat, t_norm]) 
    h_ex = exact_res[0].flatten()
    u_ex = exact_res[1].flatten()
    
    # Get NN Prediction (Physical)
    preds = bayes_nn.mean_and_std(inputs)
    h_nn = preds["sol_NN"][:, 0]
    u_nn = preds["sol_NN"][:, 1]
    
    # Squared Error
    err_h = (h_nn - h_ex)**2
    err_u = (u_nn - u_ex)**2
    data_loss_map = err_h + err_u # Total Squared Error
    
    data_loss_grid = data_loss_map.reshape(Nt, Nx)
    
    # Plotting
    print("Plotting...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. PDE Loss Distribution (Log Scale)
    ax1 = axes[0, 0]
    p1 = ax1.pcolor(X, T_mesh, np.log10(pde_loss_map + 1e-16), cmap='viridis', shading='auto')
    fig.colorbar(p1, ax=ax1, label='Log10(PDE Residual^2)')
    ax1.set_title('PDE Residual Squared (Log Scale)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('t (s)')
    # ax1.invert_yaxis() # Removed to keep (0,0) at bottom-left
    
    # 2. Data Mismatch Distribution (Log Scale)
    ax2 = axes[0, 1]
    p2 = ax2.pcolor(X, T_mesh, np.log10(data_loss_grid + 1e-16), cmap='inferno', shading='auto')
    fig.colorbar(p2, ax=ax2, label='Log10(Data Error^2)')
    ax2.set_title('Data Mismatch Squared (Log Scale)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('t (s)')
    # ax2.invert_yaxis()
    
    # 3. PDE vs Data Dominance
    ax3 = axes[1, 0]
    ratio = np.log10((pde_loss_map + 1e-16) / (data_loss_grid + 1e-16))
    p3 = ax3.pcolor(X, T_mesh, ratio, cmap='RdBu_r', vmin=-2, vmax=2, shading='auto')
    fig.colorbar(p3, ax=ax3, label='Log Ratio (PDE/Data)')
    ax3.set_title('Loss Dominance (Red=PDE, Blue=Data)')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('t (s)')
    # ax3.invert_yaxis()
    
    # 4. Weighted Loss Map
    w_pde = 1.0 / (2 * 0.005**2)
    w_data = 1.0 / (2 * 2.0**2) 
    
    weighted_map = w_pde * pde_loss_map + w_data * data_loss_grid
    
    ax4 = axes[1, 1]
    p4 = ax4.pcolor(X, T_mesh, np.log10(weighted_map + 1e-16), cmap='magma', shading='auto')
    fig.colorbar(p4, ax=ax4, label='Log10(Weighted Total Loss)')
    ax4.set_title('Weighted Total Loss Density')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('t (s)')
    # ax4.invert_yaxis()
    
    plt.tight_layout()
    save_file = os.path.join(plot_path, "loss_distribution_map.png")
    plt.savefig(save_file)
    print(f"Plot saved to {save_file}")

if __name__ == "__main__":
    plot_loss_distribution()