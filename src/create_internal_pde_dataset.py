import numpy as np
import os
import sys
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc

def create_internal_pde_dataset():
    # 1. Setup Paths
    problem = "SaintVenant1D"
    case_name = "internal_pde"
    
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

    # 2. Load Exact Solution (Same logic as before)
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
        
        L = 10000.0
        T = 14400.0
        
        t_norm_grid = t_grid_crop / T
        x_norm_grid = x_grid / L
        
        interp_h = RegularGridInterpolator((t_norm_grid, x_norm_grid), h_hist_crop, bounds_error=False, fill_value=None)
        interp_u = RegularGridInterpolator((t_norm_grid, x_norm_grid), u_hist_crop, bounds_error=False, fill_value=None)
        
        def get_exact(x_norm, t_norm):
            pts = np.stack((t_norm, x_norm), axis=1)
            h = interp_h(pts)
            u = interp_u(pts)
            return np.stack([h, u], axis=1)

    except Exception as e:
        print(f"Error loading raw data: {e}. Using dummy.")
        def get_exact(x_norm, t_norm): return np.zeros((len(x_norm), 2))

    # 3. Generate Data
    
    # A. Empty Internal Solutions (Unsupervised)
    np.save(os.path.join(save_path, "dom_sol.npy"), np.zeros((0, 2)))
    np.save(os.path.join(save_path, "sol_train.npy"), np.zeros((0, 2)))
    np.save(os.path.join(save_path, "dom_par.npy"), np.zeros((0, 2)))
    np.save(os.path.join(save_path, "par_train.npy"), np.zeros((0, 0)))
    
    # B. Boundary Data
    n_bnd = 512
    t_vals = np.linspace(0, 1, n_bnd)
    bnd_left = np.stack([np.zeros(n_bnd), t_vals], axis=1)
    bnd_right = np.stack([np.ones(n_bnd), t_vals], axis=1)
    x_vals = np.linspace(0, 1, n_bnd)
    bnd_init = np.stack([x_vals, np.zeros(n_bnd)], axis=1)
    
    dom_bnd = np.concatenate([bnd_left, bnd_right, bnd_init], axis=0)
    sol_bnd = get_exact(dom_bnd[:, 0], dom_bnd[:, 1])
    
    np.save(os.path.join(save_path, "dom_bnd.npy"), dom_bnd)
    np.save(os.path.join(save_path, "sol_bnd.npy"), sol_bnd)
    
    # C. PDE Points - FULL DOMAIN (Sobol Sequence)
    n_pde = 10000
    sampler = qmc.Sobol(d=2, scramble=True)
    dom_pde = sampler.random(n_pde) # [0,1]^2
    
    # Add boundary points to PDE set too for stability
    dom_pde = np.concatenate([dom_pde, dom_bnd], axis=0)
    
    np.save(os.path.join(save_path, "dom_pde.npy"), dom_pde)
    print(f"Generated {len(dom_pde)} PDE points (Internal + Boundary).")
    
    # D. Test Data
    nx_test = 128; nt_test = 128
    x_line = np.linspace(0, 1, nx_test)
    t_line = np.linspace(0, 1, nt_test)
    X_test, T_test = np.meshgrid(x_line, t_line)
    dom_test = np.stack([X_test.flatten(), T_test.flatten()], axis=1)
    sol_test = get_exact(dom_test[:, 0], dom_test[:, 1])
    par_test = np.zeros((len(sol_test), 0))
    
    np.save(os.path.join(save_path, "dom_test.npy"), dom_test)
    np.save(os.path.join(save_path, "sol_test.npy"), sol_test)
    np.save(os.path.join(save_path, "par_test.npy"), par_test)
    
    print("Dataset generation complete.")

if __name__ == "__main__":
    create_internal_pde_dataset()
