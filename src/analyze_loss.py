import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))
from utility import set_directory

def analyze_loss():
    set_directory()
    
    # Use the specific folder provided by the user
    path_folder = "../outs/SaintVenant1D/boundary_pde/ADAM_2025.12.01-11.47.32"
    
    if not os.path.exists(path_folder):
        print(f"Path not found: {path_folder}")
        # Fallback search logic (optional, or just return)
        return

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