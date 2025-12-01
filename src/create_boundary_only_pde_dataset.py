import numpy as np
import os
import sys
from scipy.interpolate import RegularGridInterpolator

# Manually define the generation logic to avoid threading/GUI issues in DataGenerator

def create_boundary_pde_dataset_manual():
    # 1. Setup Paths
    problem = "SaintVenant1D"
    case_name = "boundary_pde"
    
    base_dir = os.path.join("data", problem)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    save_path = os.path.join(base_dir, case_name)
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    
    # Create .gitkeep
    with open(os.path.join(save_path, ".gitkeep"), 'w') as f:
        pass
        
    print(f"Generating dataset in: {save_path}")

    # 2. Load Exact Solution for Ground Truth (Bnd and Test)
    # Re-implementing logic from SaintVenant1D_simple.values
    path_raw = "data_raw"
    try:
        h_hist = np.load(os.path.join(path_raw, "h_history.npy"))
        u_hist = np.load(os.path.join(path_raw, "u_history.npy"))
        t_grid = np.load(os.path.join(path_raw, "t_grid.npy"))
        x_grid = np.load(os.path.join(path_raw, "x_grid.npy"))
        config = np.load(os.path.join(path_raw, "config.npy"), allow_pickle=True).item()
        
        T_warmup = config.get("T_warmup", 3600.0)
        start_idx = np.searchsorted(t_grid, T_warmup)
        
        t_grid_crop = t_grid[start_idx:] - T_warmup
        h_hist_crop = h_hist[start_idx:, :]
        u_hist_crop = u_hist[start_idx:, :]
        
        # Physical Parameters
        L = 10000.0
        T = 14400.0 # Physics time
        
        # Interpolators
        # t_norm in [0, 1], x_norm in [0, 1]
        # data grid: t_grid_crop [0, ~14400], x_grid [0, 10000]
        t_norm_grid = t_grid_crop / T
        x_norm_grid = x_grid / L
        
        interp_h = RegularGridInterpolator((t_norm_grid, x_norm_grid), h_hist_crop, bounds_error=False, fill_value=None)
        interp_u = RegularGridInterpolator((t_norm_grid, x_norm_grid), u_hist_crop, bounds_error=False, fill_value=None)
        
        def get_exact(x_norm, t_norm):
            # x_norm, t_norm are 1D arrays of same length
            pts = np.stack((t_norm, x_norm), axis=1)
            h = interp_h(pts)
            u = interp_u(pts)
            return np.stack([h, u], axis=1) # (N, 2)

    except Exception as e:
        print(f"Error loading raw data: {e}. Using dummy data.")
        def get_exact(x_norm, t_norm):
            return np.zeros((len(x_norm), 2))

    # 3. Generate Data
    
    # A. Empty Internal Solutions (sol & par)
    np.save(os.path.join(save_path, "dom_sol.npy"), np.zeros((0, 2)))
    np.save(os.path.join(save_path, "sol_train.npy"), np.zeros((0, 2)))
    np.save(os.path.join(save_path, "dom_par.npy"), np.zeros((0, 2)))
    np.save(os.path.join(save_path, "par_train.npy"), np.zeros((0, 2)))
    
    # B. Boundary Data (dom_bnd, sol_bnd)
    # x=0, x=1 for t in [0,1]. Initial condition t=0 usually treated as boundary in PINNs
    n_bnd = 512
    t_vals = np.linspace(0, 1, n_bnd)
    
    # Left (x=0)
    bnd_left = np.stack([np.zeros(n_bnd), t_vals], axis=1)
    # Right (x=1)
    bnd_right = np.stack([np.ones(n_bnd), t_vals], axis=1)
    # Initial (t=0)
    x_vals = np.linspace(0, 1, n_bnd)
    bnd_init = np.stack([x_vals, np.zeros(n_bnd)], axis=1)
    
    dom_bnd = np.concatenate([bnd_left, bnd_right, bnd_init], axis=0)
    sol_bnd = get_exact(dom_bnd[:, 0], dom_bnd[:, 1])
    
    np.save(os.path.join(save_path, "dom_bnd.npy"), dom_bnd)
    np.save(os.path.join(save_path, "sol_bnd.npy"), sol_bnd)
    
    # C. PDE Points (dom_pde) - RESTRICTED TO BOUNDARY
    # We reuse dom_bnd points, maybe denser
    n_pde = 2000 # Total PDE points
    # Resample or generate new
    t_pde = np.random.uniform(0, 1, n_pde // 3)
    x_pde_init = np.random.uniform(0, 1, n_pde // 3)
    
    pde_left = np.stack([np.zeros_like(t_pde), t_pde], axis=1)
    pde_right = np.stack([np.ones_like(t_pde), t_pde], axis=1)
    pde_init = np.stack([x_pde_init, np.zeros_like(x_pde_init)], axis=1)
    
    dom_pde = np.concatenate([pde_left, pde_right, pde_init], axis=0)
    np.random.shuffle(dom_pde)
    
    np.save(os.path.join(save_path, "dom_pde.npy"), dom_pde)
    print(f"Generated {len(dom_pde)} PDE points on the boundary.")
    
    # D. Test Data (dom_test, sol_test, par_test)
    # Grid for plotting
    nx_test = 128
    nt_test = 128
    x_line = np.linspace(0, 1, nx_test)
    t_line = np.linspace(0, 1, nt_test)
    X_test, T_test = np.meshgrid(x_line, t_line)
    
    dom_test = np.stack([X_test.flatten(), T_test.flatten()], axis=1)
    sol_test = get_exact(dom_test[:, 0], dom_test[:, 1])
    par_test = np.zeros_like(sol_test) # Dummy
    
    np.save(os.path.join(save_path, "dom_test.npy"), dom_test)
    np.save(os.path.join(save_path, "sol_test.npy"), sol_test)
    np.save(os.path.join(save_path, "par_test.npy"), par_test)
    
    print("Manual dataset generation complete.")

if __name__ == "__main__":
    create_boundary_pde_dataset_manual()