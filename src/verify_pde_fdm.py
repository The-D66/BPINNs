import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import tensorflow as tf

# Add src to path
sys.path.append(os.path.dirname(__file__))
from utility import set_directory, load_json, switch_dataset, switch_equation
from setup import Parser, Param
from networks import BayesNN
from networks.Theta import Theta

def verify_pde():
    set_directory()
    
    # 1. Find latest HMC run
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
    print(f"Analyzing results in: {path_folder}")
    
    # 2. Setup Model
    config_file = "best_models/HMC_sv_1d_short"
    config_data = load_json(config_file)
    
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
    # Manually switch dataset to get config, but don't load data yet
    params.data_config = switch_dataset(params.problem, params.case_name)
    
    equation = switch_equation(params.problem)
    bayes_nn = BayesNN(params, equation)
    
    # Load Weights
    path_thetas = os.path.join(path_folder, "thetas")
    theta_subfolders = [f for f in os.listdir(path_thetas) if f.startswith("theta_")]
    if not theta_subfolders:
        print("No theta checkpoints found.")
        return
        
    latest_theta_folder = sorted(theta_subfolders)[-1]
    target_folder = os.path.join(path_thetas, latest_theta_folder)
    
    def load_list(p, name):
        l = []
        files = sorted([f for f in os.listdir(p) if f.startswith(name + "_")])
        for f in files: l.append(np.load(os.path.join(p, f)))
        return l

    ws = load_list(target_folder, "w")
    bs = load_list(target_folder, "b")
    theta_vals = []
    for w, b in zip(ws, bs):
        theta_vals.append(tf.convert_to_tensor(w, dtype=tf.float32))
        theta_vals.append(tf.convert_to_tensor(b, dtype=tf.float32))
    bayes_nn.nn_params = Theta(theta_vals)
    
    # 3. Load Data
    possible_paths = [
        "../data/SaintVenant1D/simple/dom_test.npy",
        "../data/SaintVenant1D/SaintVenant1D/dom_test.npy",
        "../data/SaintVenant1D/SaintVenant1D_simple/dom_test.npy"
    ]
    path_data_test = None
    for p in possible_paths:
        if os.path.exists(p):
            path_data_test = p
            break
    if not path_data_test:
        print("dom_test.npy not found.")
        return
        
    # Load Coordinates (Normalized)
    dom_test = np.load(path_data_test)
    # Load Ground Truth Solution (for norm coeffs)
    path_sol_test = path_data_test.replace("dom_test.npy", "sol_test.npy")
    sol_test = np.load(path_sol_test, allow_pickle=True)
    if sol_test.ndim == 0: sol_test = sol_test.item()
    
    # Set Norm Coeffs based on Test Data
    norm_mean = np.mean(sol_test, axis=0)
    norm_std = np.std(sol_test, axis=0)
    bayes_nn.norm_coeff = {"sol_mean": norm_mean, "sol_std": norm_std}
    
    # 4. Compute AD Residuals (PINN Loss Definition)
    print("Computing AD Residuals on Test Grid...")
    inputs = tf.convert_to_tensor(dom_test, dtype=tf.float32)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        u_out, f_out = bayes_nn.forward(inputs)
        # comp_residual returns combined residuals [res_cont, res_mom]
        # SaintVenant.comp_residual returns concatenation
        residuals = bayes_nn.pinn.comp_residual(inputs, u_out, f_out, tape)
    
    # residuals is (N, 2). 
    # Calculate MSE Loss (Scalar)
    mse_per_point = tf.reduce_sum(tf.square(residuals), axis=1)
    pde_loss_mse = tf.reduce_mean(mse_per_point).numpy()
    
    # Separate Cont/Mom MSE
    res_cont = residuals[:, 0].numpy()
    res_mom = residuals[:, 1].numpy()
    mse_cont = np.mean(res_cont**2)
    mse_mom = np.mean(res_mom**2)
    
    # 5. Compute FDM Residuals (For Comparison)
    print("Computing FDM Residuals...")
    # Reconstruct Grid
    L = 10000.0
    T = 14400.0
    x_flat = dom_test[:, 0] * L
    t_flat = dom_test[:, 1] * T
    
    # Prediction Denormalized
    sol_pred_norm = u_out.numpy()
    h_pred = sol_pred_norm[:, 0] * norm_std[0] + norm_mean[0]
    u_pred = sol_pred_norm[:, 1] * norm_std[1] + norm_mean[1]
    
    # Ground Truth Denormalized (it is already physical in file usually)
    # Let's check if sol_test is physical or normalized. Usually data files are physical.
    h_true = sol_test[:, 0]
    u_true = sol_test[:, 1]
    
    # Sort Grid
    sort_idx = np.lexsort((x_flat, t_flat))
    Nx = 128; Nt = 128
    X = x_flat[sort_idx].reshape(Nt, Nx)
    T_grid = t_flat[sort_idx].reshape(Nt, Nx)
    
    def calc_fdm(H, U):
        dt = np.gradient(T_grid, axis=0); dt[dt==0]=1e-8
        dx = np.gradient(X, axis=1); dx[dx==0]=1e-8
        h_t = np.gradient(H, axis=0)/dt
        h_x = np.gradient(H, axis=1)/dx
        u_t = np.gradient(U, axis=0)/dt
        u_x = np.gradient(U, axis=1)/dx
        h_safe = np.maximum(H, 0.1)
        Sf = (0.03**2 * U * np.abs(U)) / (h_safe**(4/3))
        rc = h_t + h_x * U + H * u_x
        rm = u_t + U * u_x + 9.81 * h_x + 9.81 * (Sf - 0.001)
        # Mask boundaries
        mask = np.ones_like(rc, dtype=bool)
        mask[0,:]=False; mask[-1,:]=False; mask[:,0]=False; mask[:,-1]=False
        return rc[mask], rm[mask]

    rc_p_fdm, rm_p_fdm = calc_fdm(h_pred[sort_idx].reshape(Nt,Nx), u_pred[sort_idx].reshape(Nt,Nx))
    rc_t_fdm, rm_t_fdm = calc_fdm(h_true[sort_idx].reshape(Nt,Nx), u_true[sort_idx].reshape(Nt,Nx))
    
    mse_cont_p_fdm = np.mean(rc_p_fdm**2)
    mse_mom_p_fdm = np.mean(rm_p_fdm**2)
    mse_cont_t_fdm = np.mean(rc_t_fdm**2)
    mse_mom_t_fdm = np.mean(rm_t_fdm**2)
    
    print("\n" + "="*80)
    print(f"{'{Type}':<20} | { '{Method}':<10} | { '{MSE Continuity}':<15} | { '{MSE Momentum}':<15} | { '{Total MSE}':<15}")
    print("-" * 80)
    print(f"{'{Ground Truth}':<20} | { '{FDM}':<10} | {mse_cont_t_fdm:.4e}       | {mse_mom_t_fdm:.4e}       | {mse_cont_t_fdm+mse_mom_t_fdm:.4e}")
    print(f"{'{PINN Prediction}':<20} | { '{FDM}':<10} | {mse_cont_p_fdm:.4e}       | {mse_mom_p_fdm:.4e}       | {mse_cont_p_fdm+mse_mom_p_fdm:.4e}")
    print(f"{'{PINN Prediction}':<20} | { '{AD}':<10} | {mse_cont:.4e}       | {mse_mom:.4e}       | {pde_loss_mse:.4e}")
    print("="*80)
    
    print(f"\nRatio (PINN AD / PINN FDM): {pde_loss_mse / (mse_cont_p_fdm+mse_mom_p_fdm):.2f}x")
    print(f"Ratio (PINN FDM / GT FDM):  {(mse_cont_p_fdm+mse_mom_p_fdm) / (mse_cont_t_fdm+mse_mom_t_fdm):.2f}x")

if __name__ == "__main__":
    verify_pde()