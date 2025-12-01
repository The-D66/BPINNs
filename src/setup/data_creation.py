import os
import numpy as np

class Dataset:
    def __init__(self, par, add_noise=True):
        self.pde_type = par.pde
        self.problem  = par.problem
        self.name_example = par.folder_name

        self.num_points  = par.num_points
        self.uncertainty = par.uncertainty
        self.add_noise = add_noise # New parameter

        np.random.seed(par.utils["random_seed"])

        self.__load_dataset() 
        self.__compute_norm_coeff()
        if self.add_noise: # Only add noise if add_noise is True
            self.__add_noise()

    def reload(self, add_noise=True):
        """ Reload dataset with option to toggle noise """
        self.add_noise = add_noise
        self.__load_dataset() 
        self.__compute_norm_coeff()
        if self.add_noise:
            self.__add_noise()

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
            self.__data_all[key] = (self.__data_all[key] - mean) / std

    def denormalize_dataset(self):
        for key in self.data_all:
            if key.startswith("dom"): continue
            mean, std = self.norm_coeff[f"{key[:3]}_mean"], self.norm_coeff[f"{key[:3]}_std"]
            self.__data_all[key] = self.__data_all[key] * std + mean

    def __compute_norm_coeff(self):
        self.norm_coeff = dict()
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