import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

# Add src to path
sys.path.append(os.path.dirname(__file__))
from utility import set_directory, load_json, switch_dataset
from setup import Param, Dataset

def plot_data_distribution():
    set_directory()
    
    # 1. Find latest HMC run
    parent_path = "../outs/SaintVenant1D/SaintVenant1D"
    folders = [os.path.join(parent_path, f) for f in os.listdir(parent_path) if f.startswith("HMC")]
    folders = [f for f in folders if os.path.isdir(f) and not f.endswith("/HMC") and not f.endswith(os.sep + "HMC")]
    
    if not folders:
        print("No HMC output found.")
        return
        
    path_folder = sorted(folders, key=os.path.getmtime)[-1]
    print(f"Checking data distribution for: {path_folder}")
    
    # Load Config
    config_file = "best_models/HMC_sv_1d_short"
    config_data = load_json(config_file)
    
    # Create dummy args
    class Args:
        def __init__(self):
            self.config = config_file
            self.problem = None
            self.case_name = None
            self.method = None
            self.epochs = None
            self.save_flag = False
            self.gen_flag = False # IMPORTANT: Do not regenerate, load what was used
            self.debug_flag = False
            self.random_seed = 50 # Seed used in Run 6
    args = Args()
    
    params = Param(config_data, args)
    params.data_config = switch_dataset(params.problem, params.case_name)
    
    # Load Dataset
    # Note: gen_flag=False means it will load from disk.
    # We hope the data on disk is the one used for Run 6.
    dataset = Dataset(params)
    
    # Plot
    sol_dom = dataset.data_sol["dom"] # (N, 2) normalized
    bnd_dom = dataset.data_bnd["dom"]
    
    L = 10000.0
    T = 14400.0
    
    plt.figure(figsize=(10, 8))
    
    # Plot Solution Points
    plt.scatter(sol_dom[:, 0] * L, sol_dom[:, 1] * T, s=10, c='blue', alpha=0.5, label='Solution Points (Internal)')
    
    # Plot Boundary Points
    plt.scatter(bnd_dom[:, 0] * L, bnd_dom[:, 1] * T, s=10, c='red', alpha=0.5, label='Boundary Points')
    
    # Overlay Wave Peak trajectory if possible (approx)
    # Just plotting points distribution
    
    plt.xlabel("x (m)")
    plt.ylabel("t (s)")
    plt.title(f"Training Data Distribution (Seed {args.random_seed})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(path_folder, "training_data_check.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    print(f"Number of Sol Points: {sol_dom.shape[0]}")
    print(f"Number of Bnd Points: {bnd_dom.shape[0]}")

if __name__ == "__main__":
    plot_data_distribution()
