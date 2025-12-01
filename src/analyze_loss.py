import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))
from utility import set_directory

def analyze_loss():
    set_directory()
    
    # Find latest HMC run
    # Always search in parent directory for timestamped folders
    parent_path = "../outs/SaintVenant1D/SaintVenant1D"
    
    if not os.path.exists(parent_path):
        print(f"Parent path not found: {parent_path}")
        return

    # Get full paths and filter only directories starting with HMC
    folders = [os.path.join(parent_path, f) for f in os.listdir(parent_path) if f.startswith("HMC")]
    folders = [f for f in folders if os.path.isdir(f) and not f.endswith("/HMC") and not f.endswith(os.sep + "HMC")]
    
    if not folders:
        print("No HMC output found.")
        return
        
    # Get newest folder by modification time
    latest_folder = sorted(folders, key=os.path.getmtime)[-1]
    path_folder = latest_folder
    print(f"Detected latest folder: {path_folder}")
    print(f"Analyzing results in: {path_folder}")
    
    path_log = os.path.join(path_folder, "log")
    if not os.path.exists(path_log):
        print("Log directory not found.")
        return

    # Load Keys
    keys = []
    with open(os.path.join(path_log, "keys.txt"), 'r') as f:
        for line in f: keys.append(line.strip())
    
    print(f"Loss Keys: {keys}")
    
    # Load Data
    # posterior.npy contains the RAW MSE (or loss term) values
    # loglikelihood.npy contains the weighted log-likelihood values used for optimization
    try:
        posterior = np.load(os.path.join(path_log, "posterior.npy"))
        loglikelihood = np.load(os.path.join(path_log, "loglikelihood.npy"))
    except Exception as e:
        print(f"Error loading npy files: {e}")
        return

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Raw MSE/Loss Terms (Posterior)
    ax = axes[0]
    x = np.arange(posterior.shape[1])
    
    for i, key in enumerate(keys):
        if key == "Total": continue
        vals = posterior[i, :]
        ax.plot(x, vals, label=f"{key} (MSE)", alpha=0.7)
        print(f"{key} (MSE) - Last: {vals[-1]:.2e}, Mean: {np.mean(vals):.2e}, Std: {np.std(vals):.2e}")

    ax.set_yscale('log')
    ax.set_title("Raw Loss Terms (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 2: Weighted Log-Likelihood Contributions
    # This shows what the optimizer actually sees (balancing weights)
    ax = axes[1]
    
    for i, key in enumerate(keys):
        if key == "Total": continue
        vals = loglikelihood[i, :]
        ax.plot(x, vals, label=f"{key} (LogLikelihood)", alpha=0.7)
        print(f"{key} (LLK) - Last: {vals[-1]:.2e}, Mean: {np.mean(vals):.2e}, Std: {np.std(vals):.2e}")
        
    ax.set_title("Weighted Log-Likelihood Contributions (Optimization Target)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log-Likelihood Value")
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(path_folder, "loss_analysis.png")
    plt.savefig(save_path)
    print(f"Analysis plot saved to: {save_path}")

if __name__ == "__main__":
    analyze_loss()