import os
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, par, add_noise=True):
        self.pde_type = par.pde
        self.problem  = par.problem
        self.name_example = par.folder_name

        self.num_points  = par.num_points
        self.uncertainty = par.uncertainty
        self.add_noise = add_noise 
        self.operator_mode = par.architecture.get("operator_mode", False)

        np.random.seed(par.utils["random_seed"])

        if self.operator_mode:
            self.__load_operator_dataset()
        else:
            self.__load_dataset()
            
        self.__compute_norm_coeff()
        if self.add_noise:
            self.__add_noise()

    def reload(self, add_noise=True):
        """ Reload dataset with option to toggle noise """
        self.add_noise = add_noise
        if self.operator_mode:
            self.__load_operator_dataset()
        else:
            self.__load_dataset() 
        self.__compute_norm_coeff()
        if self.add_noise:
            self.__add_noise()

    def resolve_operator_inputs(self, dom_indices):
        """ Resolve index-based 'dom' to actual OperatorNN inputs """
        # dom_indices: (N, 3) -> [Case_ID, t_val, x_val]
        
        # Ensure input is tensor
        dom = tf.convert_to_tensor(dom_indices, dtype=tf.float32)
        case_ids = tf.cast(dom[:, 0], tf.int32)
        
        # Retrieve BC/IC
        bc_batch = tf.gather(self.raw_bc, case_ids)
        ic_batch = tf.gather(self.raw_ic, case_ids)
        
        # Construct Query [x, t] (Swap t, x from dom columns 1, 2)
        # Normalize to [0, 1]
        x_raw = dom[:, 2]
        t_raw = dom[:, 1]
        
        x_min, x_max = self.x_bounds
        t_min, t_max = self.t_bounds
        
        x_norm = (x_raw - x_min) / (x_max - x_min + 1e-9)
        t_norm = (t_raw - t_min) / (t_max - t_min + 1e-9)
        
        xt_query = tf.stack([x_norm, t_norm], axis=1)
        
        return [bc_batch, ic_batch, xt_query]

    def __load_operator_dataset(self):
        """ Load parametric dataset for Operator Learning """
        # Path: ../data/SaintVenant1D/parametric_batch
        # We override self.name_example for this mode or assume it's set correctly in config
        self.path = os.path.join("../data", self.problem, "parametric_batch")
        if not os.path.exists(self.path):
            raise Exception(f"Operator dataset not found at {self.path}")

        print(f"Loading Operator Dataset from {self.path}...")
        bc_data = np.load(os.path.join(self.path, "bc_data.npy")).astype(np.float32) # (N, T, 4)
        ic_data = np.load(os.path.join(self.path, "ic_data.npy")).astype(np.float32) # (N, X, 2)
        field_data = np.load(os.path.join(self.path, "field_data.npy")).astype(np.float32) # (N, T, X, 2)
        x_grid = np.load(os.path.join(self.path, "x_grid.npy")).astype(np.float32)
        t_grid = np.load(os.path.join(self.path, "t_grid.npy")).astype(np.float32)

        self.raw_bc = bc_data
        self.raw_ic = ic_data
        
        self.x_bounds = (x_grid.min(), x_grid.max())
        self.t_bounds = (t_grid.min(), t_grid.max())
        
        N, Nt, Nx, _ = field_data.shape
        
        # Create indices for all points: (Case, Time, Space)
        # Meshgrid of indices
        case_ids = np.arange(N)
        t_ids = np.arange(Nt)
        x_ids = np.arange(Nx)
        
        # We want a flat list of all combinations
        # indexing='ij' -> (N, Nt, Nx)
        C, T, X = np.meshgrid(case_ids, t_ids, x_ids, indexing='ij')
        
        # Flatten
        C_flat = C.flatten()
        T_flat = T.flatten()
        X_flat = X.flatten()
        
        # Stack into (Total_Points, 3)
        # dom will store INDICES: [Case_ID, T_idx, X_idx]
        # We also need physical coordinates for PDE loss? 
        # Actually, for indices-based loading, we'll resolve coords later.
        # But existing LossNN expects 'dom' to be inputs to the network.
        # OperatorNN inputs are [bc, ic, query].
        # We can't pass [bc, ic] through 'dom' easily without memory explosion.
        
        # COMPROMISE: 
        # We store 'dom' as [Case_ID, t_val, x_val]. 
        # We will need a custom Collate_fn or Batcher that replaces Case_ID with actual BC/IC tensors.
        
        t_vals = t_grid[T_flat]
        x_vals = x_grid[X_flat]
        
        dom_indices = np.stack([C_flat.astype(np.float32), t_vals, x_vals], axis=1) # (Total, 3)
        
        # Solution labels
        # field_data is (N, Nt, Nx, 2)
        # flatten to (Total, 2)
        sol_flat = field_data.reshape(-1, 2)
        
        # Assign to data_sol
        # We treat ALL points as 'sol' points for Operator Learning (Supervised)
        self.__data_all = {}
        self.__data_all["dom_sol"] = dom_indices
        self.__data_all["sol_train"] = sol_flat
        
        # For PDE loss, we can use the same points or random sampling
        # Let's reuse them for now
        self.__data_all["dom_pde"] = dom_indices
        
        # Boundary/Par unused
        self.__data_all["dom_bnd"] = np.zeros((0, 3))
        self.__data_all["sol_bnd"] = np.zeros((0, 2))
        self.__data_all["dom_par"] = np.zeros((0, 3))
        self.__data_all["par_train"] = np.zeros((0, 0))
        
        # Test Data (Use last few cases as test?)
        # Or just reuse for simplicity now.
        self.__data_all["dom_test"] = dom_indices[:1000] # Sample
        self.__data_all["sol_test"] = sol_flat[:1000]
        self.__data_all["par_test"] = np.zeros((1000, 0))

    def __load_dataset(self):
        self.path = os.path.join("../data", self.problem)
        self.path = os.path.join(self.path, self.name_example)
        load = lambda name : np.load(os.path.join(self.path,name), allow_pickle=True)
        self.__data_all = {name[:-4]: load(name) for name in os.listdir(self.path) if name.endswith(".npy")}

    @property
    def data_all(self):
        return self.__data_all

    @property
    def data_sol(self):
        selected, num = ["dom_sol","sol_train"], self.num_points["sol"]
        return {k[:3]: self.data_all[k][:num,:] for k in selected}

    @property
    def data_par(self):
        selected, num = ["dom_par","par_train"], self.num_points["par"]
        return {k[:3]: self.data_all[k][:num,:] for k in selected}

    @property
    def data_bnd(self):
        selected, num = ["dom_bnd","sol_bnd"], self.num_points["bnd"]
        return {k[:3]: self.data_all[k][:num,:] for k in selected}

    @property
    def data_pde(self):
        selected, num = ["dom_pde"], self.num_points["pde"]
        return {k[:3]: self.data_all[k][:num,:] for k in selected}

    @property
    def data_test(self):
        selected = ["dom_test","sol_test","par_test"]
        return {k[:3]: self.data_all[k] for k in selected}

    @data_all.setter
    def data_all(self, items):
        name, values = items
        self.__data_all[name] = values

    @data_sol.setter
    def data_sol(self, items): self.data_all = items
    
    @data_par.setter
    def data_par(self, items): self.data_all = items
    
    @data_bnd.setter
    def data_bnd(self, items): self.data_all = items
    
    @data_pde.setter # Unused
    def data_pde(self, items): self.data_all = items
    
    @data_test.setter # Unused
    def data_test(self, items): self.data_all = items

    @property
    def data_plot(self):
        plots = dict()
        plots["sol_ex"] = (self.data_test["dom"], self.data_test["sol"])
        plots["sol_ns"] = ( self.data_sol["dom"],  self.data_sol["sol"])
        plots["par_ex"] = (self.data_test["dom"], self.data_test["par"])
        plots["par_ns"] = ( self.data_par["dom"],  self.data_par["par"])
        plots["bnd_ns"] = ( self.data_bnd["dom"],  self.data_bnd["sol"])
        return plots

    def normalize_dataset(self):
        for key in self.data_all:
            if key.startswith("dom"): continue
            mean, std = self.norm_coeff[f"{key[:3]}_mean"], self.norm_coeff[f"{key[:3]}_std"]
            # Avoid division by zero for constant fields (std=0)
            std = np.where(std < 1e-9, 1.0, std)
            self.__data_all[key] = (self.__data_all[key] - mean) / std
            
        if self.operator_mode:
            # Normalize raw BC/IC inputs
            bc_mean, bc_std = self.norm_coeff["bc_mean"], self.norm_coeff["bc_std"]
            ic_mean, ic_std = self.norm_coeff["ic_mean"], self.norm_coeff["ic_std"]
            
            # Avoid div by zero
            bc_std = np.where(bc_std < 1e-9, 1.0, bc_std)
            ic_std = np.where(ic_std < 1e-9, 1.0, ic_std)
            
            self.raw_bc = (self.raw_bc - bc_mean) / bc_std
            self.raw_ic = (self.raw_ic - ic_mean) / ic_std
            
            print(f"DEBUG: BC Norm: Mean={bc_mean}, Std={bc_std}")
            print(f"DEBUG: IC Norm: Mean={ic_mean}, Std={ic_std}")

    def denormalize_dataset(self):
        for key in self.data_all:
            if key.startswith("dom"): continue
            mean, std = self.norm_coeff[f"{key[:3]}_mean"], self.norm_coeff[f"{key[:3]}_std"]
            std = np.where(std < 1e-9, 1.0, std)
            self.__data_all[key] = self.__data_all[key] * std + mean
            
        if self.operator_mode:
            # Denormalize raw BC/IC inputs
            bc_mean, bc_std = self.norm_coeff["bc_mean"], self.norm_coeff["bc_std"]
            ic_mean, ic_std = self.norm_coeff["ic_mean"], self.norm_coeff["ic_std"]
            
            # Avoid div by zero logic matches normalize
            bc_std = np.where(bc_std < 1e-9, 1.0, bc_std)
            ic_std = np.where(ic_std < 1e-9, 1.0, ic_std)

            self.raw_bc = self.raw_bc * bc_std + bc_mean
            self.raw_ic = self.raw_ic * ic_std + ic_mean

    def __compute_norm_coeff(self):
        self.norm_coeff = dict()
        
        if self.operator_mode:
            # Use full training set for statistics in Operator Mode to avoid bias
            # self.data_sol["sol"] accesses __data_all["sol_train"]
            sol_data = self.__data_all["sol_train"]
            self.norm_coeff["sol_mean"] = np.mean(sol_data, axis=0)
            self.norm_coeff["sol_std" ] =  np.std(sol_data, axis=0)
            
            # Par is usually empty in Operator Mode
            par_data = self.__data_all["par_train"]
            if par_data.shape[0] > 0:
                self.norm_coeff["par_mean"] = np.mean(par_data, axis=0)
                self.norm_coeff["par_std" ] =  np.std(par_data, axis=0)
            else:
                self.norm_coeff["par_mean"] = np.zeros(0)
                self.norm_coeff["par_std"] = np.ones(0)

            # BC: (N, T, 4), IC: (N, X, 2)
            # Compute mean/std across (N, T) or (N, X) for each channel
            self.norm_coeff["bc_mean"] = np.mean(self.raw_bc, axis=(0, 1))
            self.norm_coeff["bc_std"]  = np.std(self.raw_bc, axis=(0, 1))
            
            self.norm_coeff["ic_mean"] = np.mean(self.raw_ic, axis=(0, 1))
            self.norm_coeff["ic_std"]  = np.std(self.raw_ic, axis=(0, 1))
            
        else:
            self.norm_coeff["sol_mean"] = np.mean(self.data_test["sol"], axis=0)
            self.norm_coeff["sol_std" ] =  np.std(self.data_test["sol"], axis=0)
            self.norm_coeff["par_mean"] = np.mean(self.data_test["par"], axis=0)
            self.norm_coeff["par_std" ] =  np.std(self.data_test["par"], axis=0)

    def __add_noise(self):
        noise_values_h = None
        noise_values_u = None

        # Check for physical noise standard deviations
        noise_h_std_phys = self.uncertainty.get("noise_h_std_phys", None)
        noise_Q_std_phys = self.uncertainty.get("noise_Q_std_phys", None)

        if noise_h_std_phys is not None and noise_Q_std_phys is not None:
            # Apply physical noise directly (data is not yet normalized)
            h_std_phys = noise_h_std_phys
            u_std_phys = noise_Q_std_phys 
            
            # Generate noise for h and u components separately with domain masking
            def noise_func(data_array, dom_array):
                # data_array: (N, 2) [h, u]
                # dom_array: (N, dim) [x, t, ...] assumed normalized [0, 1]
                
                noise_h = np.random.normal(0, h_std_phys, data_array[:,0:1].shape)
                noise_u = np.random.normal(0, u_std_phys, data_array[:,1:2].shape)
                
                # Create mask: 0.1 < x < 0.9 AND 0.1 < t < 0.9
                # Assume dom_array columns are [x, t]
                mask = np.ones((data_array.shape[0], 1), dtype=bool)
                if dom_array is not None and dom_array.shape[1] >= 1:
                    # Check spatial bounds (x)
                    mask = mask & (dom_array[:, 0:1] > 0.1) & (dom_array[:, 0:1] < 0.9)
                    # Check temporal bounds (t) if available
                    if dom_array.shape[1] >= 2:
                        mask = mask & (dom_array[:, 1:2] > 0.1) & (dom_array[:, 1:2] < 0.9)
                
                # Apply noise only where mask is True
                noise_full = np.concatenate([noise_h, noise_u], axis=1).astype("float32")
                return data_array + noise_full * mask
            
            # Apply to sol (internal points)
            self.data_sol = ("sol_train", noise_func(self.data_sol["sol"], self.data_sol["dom"]))
            # Apply to bnd (boundary points) - Mask will likely be False for x=0/1, keeping them clean!
            self.data_bnd = ("sol_bnd"  , noise_func(self.data_bnd["sol"], self.data_bnd["dom"]))

        else: # Fallback to original noise logic if physical noise not specified (Modified to support masking if needed, but keeping simple for now)
            noise_values = lambda x,y: np.random.normal(x, y, x.shape).astype("float32") 
            self.data_sol = ("sol_train", noise_values(self.data_sol["sol"], self.uncertainty["sol"]))
            self.data_par = ("par_train", noise_values(self.data_par["par"], self.uncertainty["par"]))
            self.data_bnd = ("sol_bnd"  , noise_values(self.data_bnd["sol"], self.uncertainty["bnd"]))