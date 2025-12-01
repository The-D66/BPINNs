import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))
from utility import set_directory

def check_boundary_noise():
    set_directory()
    
    # Load latest data
    # Assuming data is in ../data/SaintVenant1D/SaintVenant1D/
    path_data = "../data/SaintVenant1D/SaintVenant1D"
    
    if not os.path.exists(path_data):
        print("Data path not found.")
        return
        
    # Load Boundary Data
    sol_bnd = np.load(os.path.join(path_data, "sol_bnd.npy"))
    dom_bnd = np.load(os.path.join(path_data, "dom_bnd.npy"))
    
    # Load Exact Function
    # We need to reconstruct exact values to compare
    # Or just plot/print and see if it looks smooth
    
    print(f"Checking Boundary Data in: {path_data}")
    print(f"Shape: {sol_bnd.shape}")
    
    # Calculate smoothness (2nd derivative)
    # Sort by time for one boundary (e.g. x=0)
    mask_inlet = dom_bnd[:, 0] < 0.01 # x=0
    h_inlet = sol_bnd[mask_inlet, 0]
    t_inlet = dom_bnd[mask_inlet, 1]
    
    sort_idx = np.argsort(t_inlet)
    h_sorted = h_inlet[sort_idx]
    
    # Calculate diff
    diffs = np.diff(h_sorted)
    
    print("\n--- Inlet (x=0) Smoothness Check ---")
    print(f"Mean Abs Diff (neighboring points): {np.mean(np.abs(diffs)):.6f}")
    print(f"Max Abs Diff: {np.max(np.abs(diffs)):.6f}")
    print(f"Standard Deviation of h: {np.std(h_sorted):.4f}")
    
    # If Mean Abs Diff is large (comparable to noise level), it's noisy.
    # For clean data, it should be very small (smooth evolution).
    
    if np.mean(np.abs(diffs)) > 0.1:
        print(">>> WARNING: Boundary data appears NOISY! <<<")
    else:
        print(">>> Boundary data appears SMOOTH/CLEAN. <<<")

if __name__ == "__main__":
    check_boundary_noise()
