import os
import numpy as np
import tensorflow as tf

class WindowDataset:
    def __init__(self, params):
        self.params = params
        self.problem = params.problem
        self.window_size = params.architecture.get("window_size", 120)
        
        # Path to parametric batch data
        self.data_path = os.path.join("data", self.problem, "parametric_batch")
        if not os.path.exists(self.data_path):
            self.data_path = os.path.join("../data", self.problem, "parametric_batch")
            
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        self.__load_data()
        self.__compute_stats()
        self.normalize()
        
        # Pre-compute indices for fast loading
        self.indices = self.__prepare_indices()

    def __load_data(self):
        print(f"Loading grid data from {self.data_path}...")
        self.field_data = np.load(os.path.join(self.data_path, "field_data.npy")).astype(np.float32)
        self.bc_data = np.load(os.path.join(self.data_path, "bc_data.npy")).astype(np.float32)
        self.N_samples, self.Nt, self.Nx, self.n_channels = self.field_data.shape
        print(f"Grid loaded: {self.N_samples} samples, {self.Nt} time steps, {self.Nx} spatial points.")

    def __compute_stats(self):
        self.mean = np.mean(self.field_data, axis=(0, 1, 2))
        self.std = np.std(self.field_data, axis=(0, 1, 2))
        self.std[self.std < 1e-6] = 1.0

    def normalize(self):
        self.field_data_norm = (self.field_data - self.mean) / self.std
        mean_4 = np.concatenate([self.mean, self.mean])
        std_4 = np.concatenate([self.std, self.std])
        self.bc_data_norm = (self.bc_data - mean_4) / std_4

    def __prepare_indices(self):
        indices = []
        max_start = self.Nt - 1 - self.window_size
        for s in range(self.N_samples):
            for t in range(max_start):
                indices.append((s, t))
        return indices

    def get_data_arrays(self, mode='train'):
        """
        Returns pre-constructed NumPy arrays for the entire dataset.
        Memory intensive but fastest for training loop.
        """
        print(f"Constructing full dataset '{mode}' in RAM...")
        
        split_idx = int(0.9 * len(self.indices))
        indices_arr = np.array(self.indices)
        
        if mode == 'train':
            np.random.shuffle(indices_arr)
            selected_indices = indices_arr[:split_idx]
        else:
            selected_indices = indices_arr[split_idx:]
            
        # Pre-allocate arrays to avoid list stacking overhead
        N = len(selected_indices)
        inputs = np.zeros((N, self.window_size, self.Nx, 3), dtype=np.float32)
        targets = np.zeros((N, self.Nx, 2), dtype=np.float32)
        bcs = np.zeros((N, 4), dtype=np.float32)
        
        # Boundary Mask (Nx, 1)
        mask_bnd = np.zeros((self.Nx, 1), dtype=np.float32)
        mask_bnd[0, 0] = 1.0
        mask_bnd[-1, 0] = 1.0
        # Broadcast mask to (T, Nx, 1)
        mask_repeated = np.tile(mask_bnd[None, :, :], (self.window_size, 1, 1))
        
        # Fill arrays
        for i, (s, t_start) in enumerate(selected_indices):
            t_end = t_start + self.window_size
            
            # Field segment (T, Nx, 2)
            seq = self.field_data_norm[s, t_start:t_end]
            
            # Fill Input
            inputs[i, :, :, :2] = seq
            inputs[i, :, :, 2:] = mask_repeated
            
            # Fill Target
            targets[i] = self.field_data_norm[s, t_end]
            
            # Fill BC
            bcs[i] = self.bc_data_norm[s, t_end]
            
        print(f"Dataset '{mode}' ready. Inputs Shape: {inputs.shape}, Size: {inputs.nbytes / 1e9:.2f} GB")
        
        return inputs, targets, bcs
