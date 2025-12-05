import os
import numpy as np
import tensorflow as tf

class WindowDataset:
    def __init__(self, params, pred_horizon=10, split_ratio=0.9):
        self.params = params
        self.problem = params.problem
        self.window_size = params.architecture.get("window_size", 120)
        self.pred_horizon = pred_horizon
        self.split_ratio = split_ratio
        
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
        # Compute stats only on training split (first split_ratio of samples)
        n_train = int(self.N_samples * self.split_ratio)
        train_data = self.field_data[:n_train]
        
        print(f"Computing normalization stats on first {n_train} samples...")
        self.mean = np.mean(train_data, axis=(0, 1, 2))
        self.std = np.std(train_data, axis=(0, 1, 2))
        self.std[self.std < 1e-6] = 1.0
        
        print(f"Mean: {self.mean}, Std: {self.std}")

    def normalize(self):
        self.field_data_norm = (self.field_data - self.mean) / self.std
        mean_4 = np.concatenate([self.mean, self.mean])
        std_4 = np.concatenate([self.std, self.std])
        self.bc_data_norm = (self.bc_data - mean_4) / std_4

    def __prepare_indices(self):
        indices = []
        # Ensure we have enough future steps for pred_horizon
        max_start = self.Nt - self.window_size - self.pred_horizon
        
        if max_start < 0:
            raise ValueError(f"Sequence length {self.Nt} is too short for window {self.window_size} + horizon {self.pred_horizon}")

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
        
        split_idx = int(self.split_ratio * len(self.indices))
        indices_arr = np.array(self.indices)
        
        if mode == 'train':
            np.random.shuffle(indices_arr)
            selected_indices = indices_arr[:split_idx]
        else:
            selected_indices = indices_arr[split_idx:]
            
        # Pre-allocate arrays to avoid list stacking overhead
        N = len(selected_indices)
        # Inputs: (N, T_win, Nx, 3)
        inputs = np.zeros((N, self.window_size, self.Nx, 3), dtype=np.float32)
        # Targets: (N, T_horizon, Nx, 2)
        targets = np.zeros((N, self.pred_horizon, self.Nx, 2), dtype=np.float32)
        # BCs: (N, T_horizon, 4)
        bcs = np.zeros((N, self.pred_horizon, 4), dtype=np.float32)
        
        # Boundary Mask (Nx, 1)
        mask_bnd = np.zeros((self.Nx, 1), dtype=np.float32)
        mask_bnd[0, 0] = 1.0
        mask_bnd[-1, 0] = 1.0
        # Broadcast mask to (T, Nx, 1)
        mask_repeated = np.tile(mask_bnd[None, :, :], (self.window_size, 1, 1))
        
        # Fill arrays
        for i, (s, t_start) in enumerate(selected_indices):
            t_end_input = t_start + self.window_size
            t_end_target = t_end_input + self.pred_horizon
            
            # Input Sequence
            seq = self.field_data_norm[s, t_start:t_end_input]
            
            inputs[i, :, :, :2] = seq
            inputs[i, :, :, 2:] = mask_repeated
            
            # Target Sequence (next pred_horizon steps)
            targets[i] = self.field_data_norm[s, t_end_input:t_end_target]
            
            # BC Sequence (next pred_horizon steps)
            bcs[i] = self.bc_data_norm[s, t_end_input:t_end_target]
            
        print(f"Dataset '{mode}' ready. Inputs Shape: {inputs.shape}, Size: {inputs.nbytes / 1e9:.2f} GB")
        
        return inputs, targets, bcs
