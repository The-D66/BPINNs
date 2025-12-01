from ..template import SaintVenant1D
from dataclasses import dataclass
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

@dataclass
class SaintVenant1D_simple(SaintVenant1D):
    name    = "SaintVenant1D_simple"
    physics = {
        "length": 10000.0,
        "time": 14400.0,    # 4 hours for PINN training
        "slope": 0.001,
        "manning": 0.03,
        "h0": 5.0,
        "Q0": 20.0
    }
    
    # Specifications on the mesh
    mesh = {
        "mesh_type": "sobol",
        "test_res": 128,
        "inner_res": 20480, # More points for convection dominated problems, matching num_points.pde
        "outer_res": 256   # Boundary points
    }
    
    # Boundaries of the domains (Normalized [0,1])
    # x is [0, 1], t is [0, 1]
    domains = {
        "sol": [[(0., 1.), (0., 1.)]], # Interior
        "par": [[(0., 1.), (0., 1.)]], # Not used
        "full": [(0., 1.), (0., 1.)]
    }
    
    @property
    def values(self):
        # Helper to load reference solution
        # data_raw is in the root folder relative to src
        path_raw = "../data_raw"
        if not os.path.exists(path_raw):
             # Fallback if data not found (should not happen if generated)
             print("Warning: Reference data not found, using dummy.")
             return self.dummy_values
             
        try:
            h_hist = np.load(os.path.join(path_raw, "h_history.npy"))
            u_hist = np.load(os.path.join(path_raw, "u_history.npy"))
            t_grid = np.load(os.path.join(path_raw, "t_grid.npy"))
            x_grid = np.load(os.path.join(path_raw, "x_grid.npy"))
            config = np.load(os.path.join(path_raw, "config.npy"), allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading data: {e}")
            return self.dummy_values

        # Crop data to exclude warmup
        T_warmup = config.get("T_warmup", 3600.0) # 1 hour warmup
        
        # Find index where t >= T_warmup
        start_idx = np.searchsorted(t_grid, T_warmup)
        
        # Sliced arrays
        t_grid_crop = t_grid[start_idx:] - T_warmup # Shift time to start at 0
        h_hist_crop = h_hist[start_idx:, :]
        u_hist_crop = u_hist[start_idx:, :]
        
        # Check if cropped time matches physics["time"] (14400s)
        # t_grid_crop should go from 0 to 14400
        
        # Normalize grid for interpolation to match network inputs [0,1]
        # We use self.physics["time"] which is 14400s
        t_norm = t_grid_crop / self.physics["time"]
        x_norm = x_grid / self.physics["length"]
        
        interp_h = RegularGridInterpolator((t_norm, x_norm), h_hist_crop, bounds_error=False, fill_value=None)
        interp_u = RegularGridInterpolator((t_norm, x_norm), u_hist_crop, bounds_error=False, fill_value=None)
        
        def exact_func(inputs):
            # inputs: [x_coords, t_coords] (2, N)
            x_in = inputs[0]
            t_in = inputs[1]
            
            # Interpolator expects (N, 2) points as (t, x)
            # Stack t and x
            pts = np.stack((t_in, x_in), axis=1)
            
            h_val = interp_h(pts)
            u_val = interp_u(pts)
            
            return [h_val, u_val]

        return {
            "u": exact_func,
            "f": lambda x: [np.sin(x[0])] # Dummy parametric field (unused)
        }

    @property
    def dummy_values(self):
        # Fallback dummy function
        h0 = self.physics["h0"]
        u0 = self.physics["Q0"] / h0
        return {
            "u": lambda x: [np.full_like(x[0], h0), np.full_like(x[0], u0)],
            "f": lambda x: [np.zeros_like(x[0])]
        }
    
    # Overwrite values with the smart boundary-aware function
    # We don't need explicit boundary_values logic anymore because `values` (FVM solution)
    # already contains the correct boundary evolution.