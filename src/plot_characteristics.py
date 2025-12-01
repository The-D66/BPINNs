import numpy as np
import matplotlib.pyplot as plt
import os

def plot_characteristics():
    path_raw = "data_raw"
    if not os.path.exists(path_raw):
        print("Data raw folder not found.")
        return

    try:
        h_hist = np.load(os.path.join(path_raw, "h_history.npy"))
        u_hist = np.load(os.path.join(path_raw, "u_history.npy"))
        t_grid = np.load(os.path.join(path_raw, "t_grid.npy"))
        x_grid = np.load(os.path.join(path_raw, "x_grid.npy"))
        config = np.load(os.path.join(path_raw, "config.npy"), allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Physics
    g = 9.81
    T_warmup = config.get("T_warmup", 3600.0)

    # Crop to warmup (PINN domain)
    start_idx = np.searchsorted(t_grid, T_warmup)
    t_crop = t_grid[start_idx:] - T_warmup
    h_crop = h_hist[start_idx:, :]
    u_crop = u_hist[start_idx:, :]
    
    # Grid for plotting
    # Meshgrid expects x (cols) and y (rows) usually, but streamplot matches array shapes
    # streamplot(x, y, u, v) where x, y are 1d or 2d.
    # Our data is (Nt, Nx). 
    # Let's create 2D meshgrids.
    X, T = np.meshgrid(x_grid, t_crop)
    
    # Calculate Wave Celerity c = sqrt(gh)
    # Ensure h is positive
    h_crop = np.maximum(h_crop, 0.0)
    c = np.sqrt(g * h_crop)
    
    # Characteristic Speeds
    # lambda = dx/dt
    # Vector field in (x, t) space: (dx, dt) = (lambda, 1)
    lambda_1 = u_crop + c # Downstream characteristic
    lambda_2 = u_crop - c # Upstream characteristic
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 16)) # 2 rows, 1 col. Tall figure.
    
    # Subplot 1: u + c
    ax1 = axes[0]
    im1 = ax1.imshow(lambda_1, aspect='auto', extent=[x_grid[0], x_grid[-1], t_crop[-1], t_crop[0]],
                     cmap='viridis') # Viridis for positive speeds
    ax1.set_title(r"Characteristic Speed $\lambda_1 = u + \sqrt{gh}$ (m/s)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("t (s)")
    fig.colorbar(im1, ax=ax1, label=r'Speed (m/s)')

    # Subplot 2: u - c
    ax2 = axes[1]
    # Use divergent colormap for speeds that can be positive or negative
    im2 = ax2.imshow(lambda_2, aspect='auto', extent=[x_grid[0], x_grid[-1], t_crop[-1], t_crop[0]],
                     cmap='RdBu_r') # Red-Blue divergent colormap
    ax2.set_title(r"Characteristic Speed $\lambda_2 = u - \sqrt{gh}$ (m/s)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("t (s)")
    fig.colorbar(im2, ax=ax2, label=r'Speed (m/s)')

    plt.tight_layout()
    save_path = "data_raw/characteristics_heatmap.png"
    plt.savefig(save_path, dpi=300)
    print(f"Characteristics heatmap plot saved to {save_path}")

if __name__ == "__main__":
    plot_characteristics()

if __name__ == "__main__":
    plot_characteristics()