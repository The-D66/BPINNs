import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))
from utility import set_directory

def inspect_failure():
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
    print(f"Inspecting results in: {path_folder}")
    
    # 2. Load Predictions
    sol_NN = np.load(os.path.join(path_folder, "values", "sol_NN.npy"))
    
    # 3. Load Ground Truth
    # Trying standard paths
    path_data_test = "../data/SaintVenant1D/SaintVenant1D/dom_test.npy"
    path_sol_test = "../data/SaintVenant1D/SaintVenant1D/sol_test.npy"
    
    if not os.path.exists(path_data_test):
        print("Data not found.")
        return
        
    dom_test = np.load(path_data_test)
    sol_test = np.load(path_sol_test, allow_pickle=True)
    if sol_test.ndim == 0: sol_test = sol_test.item()
    
    # 4. Reconstruct Grid
    L = 10000.0
    T = 14400.0
    
    x_flat = dom_test[:, 0] * L
    t_flat = dom_test[:, 1] * T
    h_pred_flat = sol_NN[:, 0]
    h_true_flat = sol_test[:, 0]
    
    sort_idx = np.lexsort((x_flat, t_flat))
    x_sorted = x_flat[sort_idx]
    t_sorted = t_flat[sort_idx]
    
    unique_t = np.unique(t_sorted)
    unique_x = np.unique(x_sorted)
    Nt = len(unique_t)
    Nx = len(unique_x)
    
    X = x_sorted.reshape(Nt, Nx)
    T_grid = t_sorted.reshape(Nt, Nx)
    H_pred = h_pred_flat[sort_idx].reshape(Nt, Nx)
    H_true = h_true_flat[sort_idx].reshape(Nt, Nx)
    
    # 5. Critical Analysis
    print("\n" + "="*60)
    print("CRITICAL FAILURE ANALYSIS (Water Depth h)")
    print("="*60)
    
    # A. Max Error (Worst Case)
    abs_diff = np.abs(H_pred - H_true)
    max_error = np.max(abs_diff)
    max_error_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    max_err_time = T_grid[max_error_idx] / 3600.0
    max_err_loc = X[max_error_idx] / 1000.0
    max_val_true = H_true[max_error_idx]
    
    print(f"1. Global Max Error:")
    print(f"   Value: {max_error:.4f} m")
    print(f"   Location: x={max_err_loc:.2f} km, t={max_err_time:.2f} h")
    print(f"   True Value there: {max_val_true:.4f} m")
    print(f"   Relative Error at worst point: {max_error/max_val_true*100:.2f}%")
    
    # B. Transient Region Analysis (First 2 hours)
    mask_transient = T_grid < 7200 # 2 hours
    mean_err_transient = np.mean(abs_diff[mask_transient])
    max_err_transient = np.max(abs_diff[mask_transient])
    
    print(f"\n2. Transient Region (t < 2h) Error:")
    print(f"   Mean Error: {mean_err_transient:.4f} m")
    print(f"   Max Error:  {max_err_transient:.4f} m")
    
    # C. Peak Analysis at x=0.0 km (Inlet Boundary)
    # Find closest index to 0km
    idx_x_bnd = np.argmin(np.abs(unique_x - 0.0))
    h_true_bnd = H_true[:, idx_x_bnd]
    h_pred_bnd = H_pred[:, idx_x_bnd]
    t_bnd = T_grid[:, idx_x_bnd] / 3600.0
    
    # Calculate Boundary Error
    mse_bnd = np.mean((h_true_bnd - h_pred_bnd)**2)
    max_err_bnd = np.max(np.abs(h_true_bnd - h_pred_bnd))
    
    print(f"\n3. Boundary Analysis at x={unique_x[idx_x_bnd]/1000:.1f} km (Inlet):")
    print(f"   MSE Error: {mse_bnd:.4e}")
    print(f"   Max Error: {max_err_bnd:.4f} m")

    # Plot the boundary comparison
    plt.figure(figsize=(10, 6))
    plt.plot(t_bnd, h_true_bnd, 'k-', linewidth=2, label='True (Inlet)')
    plt.plot(t_bnd, h_pred_bnd, 'r--', linewidth=2, label='Prediction')
    plt.title(f"Inlet Boundary Profile at x={unique_x[idx_x_bnd]/1000:.1f} km")
    plt.xlabel("Time (h)")
    plt.ylabel("h (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(path_folder, "failure_analysis_inlet.png")
    plt.savefig(save_path)
    print(f"\nInlet plot saved to: {save_path}")

if __name__ == "__main__":
    inspect_failure()
