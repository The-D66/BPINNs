class Seq2SeqTrainer:
    def __init__(self, model, params, dataset, output_dir=None):
        self.model = model
        self.params = params
        self.dataset = dataset
        
        # Use provided output dir or default
        if output_dir:
            self.ckpt_dir = os.path.join(output_dir, "checkpoints")
            self.plot_dir = os.path.join(output_dir, "plots")
        else:
            self.ckpt_dir = f"outs/{self.params.problem}/checkpoints"
            self.plot_dir = f"outs/{self.params.problem}/plots"
            
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        print(f"Outputs will be saved to: {output_dir}")
        
        # Training params
        self.epochs = params.param_method["epochs"]
        self.lr = params.param_method.get("lr", 1e-3)
        self.batch_size = params.param_method["batch_size"]
        
        self.lambda_data = params.param_method.get("lambda_data", 1.0)
        self.lambda_pde = params.param_method.get("lambda_pde", 0.1)
        self.beta_grad = 10.0
        
        # Autoregressive specific
        self.window_size = dataset.window_size
        self.pred_horizon = dataset.pred_horizon # Get from dataset
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        # PDE params
        self.dx = params.physics["dx"]
        self.dt = params.physics["dt"]
        # self.dx_kernel = tf.constant([-0.5, 0.0, 0.5], shape=(3, 1, 1), dtype=tf.float32) / self.dx # Not used anymore
        
        # Stats
        self.mean = tf.constant(dataset.mean, dtype=tf.float32)
        self.std = tf.constant(dataset.std, dtype=tf.float32)

    def denormalize(self, u_norm):
        return u_norm * self.std + self.mean

    def compute_pde_residual(self, u_next_norm, u_prev_norm):
        # Crank-Nicolson Scheme:
        # (U_new - U_old)/dt = 0.5 * (F(U_new) + F(U_old))
        
        u_next = self.denormalize(u_next_norm)
        u_prev = self.denormalize(u_prev_norm)
        
        # Time derivative
        u_t_phys = (u_next - u_prev) / self.dt
        
        h_next = u_next[..., 0:1]
        u_next_val = u_next[..., 1:2]
        
        h_prev = u_prev[..., 0:1]
        u_prev_val = u_prev[..., 1:2]
        
        # Helper to compute spatial terms F(U)
        def spatial_terms(h, u_val):
            # For gradients, need to be sure h and u are watche by tape
            # But here we are just taking finite differences.
            h_padded = tf.pad(h, [[0,0], [1,1], [0,0]], "SYMMETRIC")
            u_padded = tf.pad(u_val, [[0,0], [1,1], [0,0]], "SYMMETRIC")
            h_x = (h_padded[:, 2:, :] - h_padded[:, :-2, :]) / (2 * self.dx)
            u_x = (u_padded[:, 2:, :] - u_padded[:, :-2, :]) / (2 * self.dx)
            
            g = 9.81
            S0 = self.params.physics["slope"]
            n_manning = self.params.physics["manning"]
            h_safe = tf.maximum(h, 0.1)
            Sf = (n_manning**2 * u_val * tf.abs(u_val)) / (h_safe**(4/3))
            
            # Mass: (hu)_x = h u_x + u h_x
            term_mass_spatial = h * u_x + u_val * h_x
            # Momentum: u u_x + g h_x + g(Sf - S0)
            term_mom_spatial = u_val * u_x + g * h_x + g * (Sf - S0)
            
            return term_mass_spatial, term_mom_spatial

        # Terms at t+1
        mass_spatial_next, mom_spatial_next = spatial_terms(h_next, u_next_val)
        # Terms at t
        mass_spatial_prev, mom_spatial_prev = spatial_terms(h_prev, u_prev_val)
        
        # Average (Crank-Nicolson)
        res_mass = u_t_phys[..., 0:1] + 0.5 * (mass_spatial_next + mass_spatial_prev)
        res_mom = u_t_phys[..., 1:2] + 0.5 * (mom_spatial_next + mom_spatial_prev)
        
        return res_mass, res_mom

    def compute_gradient_weights(self, target):
        target_padded = tf.pad(target, [[0,0], [1,1], [0,0]], "SYMMETRIC")
        target_x = (target_padded[:, 2:, :] - target_padded[:, :-2, :]) / (2 * self.dx)
        grad_mag = tf.abs(target_x)
        return 1.0 + self.beta_grad * grad_mag

    @tf.function
    def train_step(self, inputs, targets, bcs):
        """
        Performs a multi-step autoregressive training step.
        inputs: (Batch, Window_Size, Nx, 3)
        targets: (Batch, Pred_Horizon, Nx, 2)
        bcs: (Batch, Pred_Horizon, 4) - Not used in current PDE residual, but passed.
        """
        current_inputs = inputs # This is the history window (B, W, Nx, 3)
        
        total_loss = 0.0
        loss_data_log = 0.0
        loss_pde_log = 0.0
        
        # Wrap the entire autoregressive loop in a single GradientTape
        with tf.GradientTape() as tape:
            for k_step in tf.range(self.pred_horizon):
                # Predict one step ahead
                # model expects (B, W, Nx, 3) -> (B, Nx, 2)
                pred_norm = self.model(current_inputs) 
                
                # Apply boundary conditions if needed (hard BCs not done by ConvFormer yet)
                # Currently, ConvFormer doesn't take bc_next, so we just use pred_norm
                
                # Extract current target for this step
                current_target = targets[:, k_step, :, :] # (B, Nx, 2)
                
                # Data Loss
                weights = self.compute_gradient_weights(current_target)
                weighted_sq_diff = weights * tf.square(pred_norm - current_target)
                loss_data_k = tf.reduce_mean(weighted_sq_diff)
                
                # PDE Loss (Crank-Nicolson)
                # u_prev_norm is the LAST frame of the current_inputs history window
                u_prev_norm_for_pde = current_inputs[:, -1, :, 0:2] 
                res_mass, res_mom = self.compute_pde_residual(pred_norm, u_prev_norm_for_pde)
                loss_pde_k = tf.reduce_mean(tf.square(res_mass)) + tf.reduce_mean(tf.square(res_mom))
                
                # Total Loss for this step
                loss_k = self.lambda_data * loss_data_k + self.lambda_pde * loss_pde_k
                
                total_loss += loss_k
                
                # For logging, we'll use the losses from the last step
                loss_data_log = loss_data_k
                loss_pde_log = loss_pde_k

                # Autoregressive Update: Append predicted frame to history, remove oldest
                # The model's input expects (B, W, Nx, 3)
                # pred_norm is (B, Nx, 2). The third channel is mask.
                # We need to preserve the mask from the last input frame.
                last_mask = current_inputs[:, -1, :, 2:] # (B, Nx, 1)
                pred_with_mask = tf.concat([pred_norm, last_mask], axis=-1) # (B, Nx, 3)
                
                # Shift the window: remove first frame, append new_pred_with_mask
                # current_inputs[:, 1:, :, :] gets all frames from 2nd to last
                current_inputs = tf.concat([current_inputs[:, 1:, :, :], tf.expand_dims(pred_with_mask, axis=1)], axis=1)
                
        # Average the total loss over the prediction horizon
        final_loss = total_loss / tf.cast(self.pred_horizon, tf.float32)
            
        grads = tape.gradient(final_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return final_loss, loss_data_log, loss_pde_log # Return last step's loss parts for logging

    def train(self):
        print(f"Starting ST-APINO Training for {self.epochs} epochs...")
        
        # Load ALL data into memory (NumPy arrays)
        # train_inputs: (N, W, Nx, 3), train_targets: (N, K, Nx, 2), train_bcs: (N, K, 4)
        train_inputs, train_targets, train_bcs = self.dataset.get_data_arrays(mode='train')
        N_train = len(train_inputs)
        steps_per_epoch = N_train // self.batch_size
        
        print(f"Train Samples: {N_train}, Batch Size: {self.batch_size}, Steps: {steps_per_epoch}")
        
        start_time = time.time()
        best_error = float('inf') # For validation

        # Ensure model is built for summary
        if not self.model.built:
            dummy_input = tf.zeros((1, self.window_size, self.dataset.Nx, 3))
            _ = self.model(dummy_input)
        self.model.summary()
        
        for epoch in range(self.epochs):
            total_loss = 0.0; total_data_loss = 0.0; total_pde_loss = 0.0
            
            # Manual Shuffle
            indices = np.arange(N_train)
            np.random.shuffle(indices)
            
            # Manual Batch Loop (No tf.data overhead)
            for i in range(0, N_train, self.batch_size):
                end_idx = min(i + self.batch_size, N_train)
                batch_idx = indices[i:end_idx]
                
                # Slicing numpy array is fast
                x_batch = tf.convert_to_tensor(train_inputs[batch_idx])
                y_batch = tf.convert_to_tensor(train_targets[batch_idx])
                bc_batch = tf.convert_to_tensor(train_bcs[batch_idx])
                
                l, l_d, l_p = self.train_step(x_batch, y_batch, bc_batch)
                
                # Accumulate scalar values (numpy conversion avoids graph memory leak)
                total_loss += l.numpy()
                total_data_loss += l_d.numpy()
                total_pde_loss += l_p.numpy()
                
            avg_loss = total_loss / steps_per_epoch
            avg_data_loss = total_data_loss / steps_per_epoch
            avg_pde_loss = total_pde_loss / steps_per_epoch
            
            # Logging
            elapsed = time.time() - start_time
            log_str = f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.6f} (D:{avg_data_loss:.6f} P:{avg_pde_loss:.6f})"
            
            # Validation
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1: # Also validate on last epoch
                val_error = validate_model(self.model, self.params, self.dataset, epoch=epoch+1, save_dir=self.plot_dir)
                log_str += f" | Val Err: {val_error:.4%}"
                
                # Save Checkpoint (epoch-wise and best)
                self.model.save_weights(os.path.join(self.ckpt_dir, f"model_epoch_{epoch+1}.weights.h5"))
                
                if val_error < best_error:
                    best_error = val_error
                    self.model.save_weights(os.path.join(self.ckpt_dir, "model_best.weights.h5"))
                    log_str += " [BEST]"
            
            print(log_str + f" | Time: {elapsed:.1f}s")