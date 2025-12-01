import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def reproduce_error_plot():
    # Define paths
    base_path = "outs/SaintVenant1D/SaintVenant1D/HMC_2025.11.25-10.29.20"
    path_data = os.path.join(base_path, "data")
    path_values = os.path.join(base_path, "values")
    save_path = os.path.join(base_path, "plot", "full_domain_error_with_characteristics.png")

    print(f"Loading data from {base_path}...")

    # Load Coordinates (Physical)
    # sol_ex_dom is (N, 2) -> [x, t]
    dom = np.load(os.path.join(path_data, "sol_ex_dom.npy"), allow_pickle=True)
    print(f"DEBUG: dom type: {type(dom)}")
    if isinstance(dom, np.ndarray):
        print(f"DEBUG: dom shape: {dom.shape}")
    else:
        print(f"DEBUG: dom content: {dom}")
        
    # Handle case where dom might be 0-d array containing a tuple/list
    if dom.shape == ():
        dom = dom.item()
        print(f"DEBUG: Extracted item, new type: {type(dom)}")
        if isinstance(dom, np.ndarray):
             print(f"DEBUG: New shape: {dom.shape}")

    x_flat = dom[:, 0]
    t_flat = dom[:, 1]

    # Load Exact Solution
    # sol_ex_val is (N, 2) -> [h, u]
    exact = np.load(os.path.join(path_data, "sol_ex_val.npy"), allow_pickle=True)
    print(f"DEBUG: exact type: {type(exact)}")
    if isinstance(exact, np.ndarray):
        print(f"DEBUG: exact shape: {exact.shape}")
        
    h_ex = exact[:, 0]
    u_ex = exact[:, 1]

    # Load Predicted Solution
    # sol_NN is (N, 2) -> [h, u]
    pred = np.load(os.path.join(path_values, "sol_NN.npy"), allow_pickle=True)
    print(f"DEBUG: pred type: {type(pred)}")
    if isinstance(pred, np.ndarray):
        print(f"DEBUG: pred shape: {pred.shape}")

    h_nn = pred[:, 0]
    u_nn = pred[:, 1]

    # Calculate Error
    err_h = h_nn - h_ex
    err_u = u_nn - u_ex
    
    rmse_h = np.sqrt(np.mean(err_h**2))
    rmse_u = np.sqrt(np.mean(err_u**2))

    # Reshape for plotting
    # Identify unique grid points
    x_unique = np.unique(x_flat)
    t_unique = np.unique(t_flat)
    Nx = len(x_unique)
    Nt = len(t_unique)
    
    print(f"Grid shape: Nx={Nx}, Nt={Nt}, Total={len(x_flat)}")
    
    # Reshape arrays to (Nt, Nx)
    # Note: We need to be careful with ordering. Usually sorting helps.
    # Assuming structured grid.
    
    # Create meshgrid for plotting
    X, T = np.meshgrid(x_unique, t_unique)
    
    # Helper to reshape
    def reshape_to_grid(flat_arr):
        # Sort by t then x to ensure correct reshaping order
        # This assumes the data is tensor product grid
        # Create a structured array to sort
        struct = np.rec.fromarrays([t_flat, x_flat, flat_arr], names=['t', 'x', 'val'])
        struct.sort(order=['t', 'x'])
        return struct['val'].reshape((Nt, Nx))

    Z_err_h = reshape_to_grid(err_h)
    Z_err_u = reshape_to_grid(err_u)
    
    # Calculate Characteristics (using Exact solution for reference)
    # c = sqrt(gh)
    g = 9.81
    h_grid = reshape_to_grid(h_ex)
    u_grid = reshape_to_grid(u_ex)
    
    # Ensure h is positive for sqrt
    h_grid = np.maximum(h_grid, 0.0)
    c_grid = np.sqrt(g * h_grid)
    
    # Characteristic speeds
    lambda1 = u_grid + c_grid # u + c
    lambda2 = u_grid - c_grid # u - c
    
    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex='col', sharey='col')
    fig.tight_layout(pad=4.0)
    
    # Limits for colorbar symmetric
    def get_limit(arr):
        m = np.max(np.abs(arr))
        return m if m > 1e-8 else 0.1
    
    lim_h = get_limit(Z_err_h)
    lim_u = get_limit(Z_err_u)

    # Downsample for Quiver
    skip_t = max(1, Nt // 20)
    skip_x = max(1, Nx // 20)
    
    X_q = X[::skip_t, ::skip_x]
    T_q = T[::skip_t, ::skip_x]
    
    # Quiver vectors - scaled to represent physical displacement
    # We want arrows to be visible and point correctly in (x, t) space.
    # Since x range ~10000 and t range ~14400, aspect ratio is roughly 1:1.4
    # But lambda ~ 5 m/s. dx/dt = 5.
    # In 1000s, dx = 5000m.
    # Vector (5000, 1000) in data coordinates.
    
    dt_arrow = 1000.0 # Draw arrows representing 1000s of travel
    l1_val = lambda1[::skip_t, ::skip_x]
    l2_val = lambda2[::skip_t, ::skip_x]
    
    U1 = l1_val * dt_arrow
    V1 = np.full_like(U1, dt_arrow)
    
    U2 = l2_val * dt_arrow
    V2 = np.full_like(U2, dt_arrow)
    
    # --- Row 1: Water Depth h ---
    
    # Map
    ax1 = axes[0, 0]
    p1 = ax1.pcolor(X, T, Z_err_h, cmap='RdBu_r', vmin=-lim_h, vmax=lim_h, shading='auto')
    cb1 = fig.colorbar(p1, ax=ax1)
    cb1.set_label("Error (m)")
    
    # Overlay Characteristics
    # Use angles='xy' so arrows point in data direction (x, t)
    # scale_units='xy' and scale=1 means vector (u, v) is plotted as (u, v) in data units?
    # Usually scale is inverse. Let's try auto-scale first but with angles='xy'
    q1 = ax1.quiver(X_q, T_q, U1, V1, color='k', alpha=0.4, width=0.003, headwidth=3, angles='xy', scale_units='xy', scale=10000, label='u+c')
    q2 = ax1.quiver(X_q, T_q, U2, V2, color='gray', alpha=0.4, width=0.003, headwidth=3, angles='xy', scale_units='xy', scale=10000, label='u-c')
    
    ax1.set_title("h Error & Characteristics (u±c)")
    ax1.set_ylabel("t (s)")
    ax1.set_xlabel("x (m)")
    ax1.invert_yaxis()

    # Histogram
    ax2 = axes[0, 1]
    ax2.hist(err_h, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Error Value (m)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Water Depth Error Histogram\nRMSE = {rmse_h:.4e} m")
    ax2.grid(True, alpha=0.3)

    # --- Row 2: Velocity u ---
    
    # Map
    ax3 = axes[1, 0]
    p2 = ax3.pcolor(X, T, Z_err_u, cmap='RdBu_r', vmin=-lim_u, vmax=lim_u, shading='auto')
    cb2 = fig.colorbar(p2, ax=ax3)
    cb2.set_label("Error (m/s)")
    
    # Overlay Characteristics (Same as above)
    ax3.quiver(X_q, T_q, U1, V1, color='k', alpha=0.4, width=0.003, headwidth=3, angles='xy', scale_units='xy', scale=10000)
    ax3.quiver(X_q, T_q, U2, V2, color='gray', alpha=0.4, width=0.003, headwidth=3, angles='xy', scale_units='xy', scale=10000)
    
    ax3.set_title("u Error & Characteristics (u±c)")
    ax3.set_ylabel("t (s)")
    ax3.set_xlabel("x (m)")
    ax3.invert_yaxis()

    # Histogram
    ax4 = axes[1, 1]
    ax4.hist(err_u, bins=50, color='salmon', edgecolor='black', alpha=0.7)
    ax4.set_xlabel("Error Value (m/s)")
    ax4.set_ylabel("Frequency")
    ax4.set_title(f"Velocity Error Histogram\nRMSE = {rmse_u:.4e} m/s")
    ax4.grid(True, alpha=0.3)

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    reproduce_error_plot()
