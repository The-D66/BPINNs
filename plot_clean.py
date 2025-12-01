
import os
# Critical Fix for macOS OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import sys
# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from postprocessing.Plotter import Plotter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
OUT_DIR = "outs/clean_solve"
L_PHYS = 10000.0
T_PHYS = 14400.0

def main():
    # Load Data
    try:
        history_loss = np.load(os.path.join(OUT_DIR, "history_loss.npy"))
        inputs_full = np.load(os.path.join(OUT_DIR, "inputs_full.npy"))
        Y_pred = np.load(os.path.join(OUT_DIR, "Y_pred.npy"))
        Y_exact = np.load(os.path.join(OUT_DIR, "Y_exact.npy"))
    except FileNotFoundError:
        print("Error: Data files not found. Run train_clean.py (or mock) first.")
        sys.exit(1)

    # Prepare Data Dictionary for Plotter
    # Plotter expects:
    # data["sol_ex"] = (coords, values)
    # coords: [x, t]
    # values: [h, u] (order matters for labels)
    
    # Y_exact is [h, u] (from train_clean.py)
    # Y_pred is [h, u]
    
    # Plotter.py logic:
    # scale_x = 10.0, scale_t = 4.0 (hardcoded for SaintVenant)
    # inputs_full is Normalized [0, 1].
    # 1.0 * 10.0 = 10km. Correct.
    
    data_plot = {
        "sol_ex": (inputs_full, Y_exact),
        "par_ex": (inputs_full, np.zeros_like(inputs_full)),
        "sol_ns": (inputs_full[::100], Y_exact[::100]), 
        "par_ns": None
    }
    
    functions_plot = {
        "sol_NN": Y_pred,
        "sol_std": np.zeros_like(Y_pred),
        "par_NN": np.zeros_like(Y_pred),
        "par_std": np.zeros_like(Y_pred)
    }
    
    # Plot
    plotter = Plotter(OUT_DIR)
    
    print("Plotting Confidence/Comparison...")
    plotter.plot_confidence(data_plot, functions_plot)
    
    print("Plotting Loss History...")
    # Plotter expects list of dicts for loss
    plotter.plot_losses([{"Total": history_loss}, {"Total": history_loss}])
    
    print("Plotting Full Domain Error...")
    plotter.plot_full_domain_error(data_plot, functions_plot)
    
    print("Plotting QQ Plot...")
    try:
        plotter.plot_qq(data_plot, functions_plot)
    except AttributeError:
        print("Plotter.plot_qq method not found (might be custom). Skipping.")
        pass
        
    print(f"Plots saved to {OUT_DIR}/plot/")

if __name__ == "__main__":
    main()
