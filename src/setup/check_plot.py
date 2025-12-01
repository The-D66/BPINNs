import numpy as np
import matplotlib.pyplot as plt
import os

def plot_generated_data():
    path_raw = "data_raw"
    if not os.path.exists(path_raw):
        print("Data not found.")
        return

    try:
        h_hist = np.load(os.path.join(path_raw, "h_history.npy"))
        u_hist = np.load(os.path.join(path_raw, "u_history.npy"))
        config = np.load(os.path.join(path_raw, "config.npy"), allow_pickle=True).item()
        
        L = config.get("L", 10000.0)
        T_total = config.get("T_total", 8400.0) # Use T_total from config if available, or infer
        T_warmup = config.get("T_warmup", 1200.0)
        
        # If T_total not in config, infer from grid or hardcode from known generator
        if "T_total" not in config:
             # Generator used T_total = T_warmup + T_sim
             # Let's assume standard params if not saved
             T_total = 1200.0 + 7200.0
        
        print(f"Plotting data: L={L}, T_total={T_total}, Warmup={T_warmup}")

        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.imshow(h_history, aspect='auto', extent=[0, L, T_total, 0], cmap='RdBu')
        plt.colorbar(label='Water Depth h (m)')
        plt.axhline(y=T_warmup, color='k', linestyle='--', label='Warmup End')
        plt.xlabel('x (m)')
        plt.ylabel('t (s)')
        plt.legend()
        plt.title('Water Depth (Full Simulation)')
        
        plt.subplot(2, 1, 2)
        plt.imshow(u_history, aspect='auto', extent=[0, L, T_total, 0], cmap='RdBu')
        plt.colorbar(label='Velocity u (m/s)')
        plt.axhline(y=T_warmup, color='k', linestyle='--', label='Warmup End')
        plt.xlabel('x (m)')
        plt.ylabel('t (s)')
        plt.legend()
        plt.title('Velocity (Full Simulation)')
        
        plt.tight_layout()
        plt.savefig("data_raw/reference_solution_check.png")
        print("Plot saved to data_raw/reference_solution_check.png")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Local variable h_history needs to be loaded before plotting
    # Re-loading here for safety
    path_raw = "data_raw"
    h_history = np.load(os.path.join(path_raw, "h_history.npy"))
    u_history = np.load(os.path.join(path_raw, "u_history.npy"))
    plot_generated_data()
