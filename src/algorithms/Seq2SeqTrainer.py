import tensorflow as tf
import numpy as np
import os
import time
from verify_rollout import validate_model

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
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        # PDE params
        self.dx = params.physics["dx"]
        self.dt = params.physics["dt"]
        self.dx_kernel = tf.constant([-0.5, 0.0, 0.5], shape=(3, 1, 1), dtype=tf.float32) / self.dx
        
        # Stats
        self.mean = tf.constant(dataset.mean, dtype=tf.float32)
        self.std = tf.constant(dataset.std, dtype=tf.float32)

    def denormalize(self, u_norm):
        return u_norm * self.std + self.mean

    def compute_pde_residual(self, u_next_norm, u_prev_norm):
        u_next = self.denormalize(u_next_norm)
        u_prev = self.denormalize(u_prev_norm)
        u_t = (u_next - u_prev) / self.dt
        
        h = u_next[..., 0:1]
        u = u_next[..., 1:2]
        h_t = u_t[..., 0:1]
        u_t_momentum = u_t[..., 1:2]
        
        h_padded = tf.pad(h, [[0,0], [1,1], [0,0]], "SYMMETRIC")
        u_padded = tf.pad(u, [[0,0], [1,1], [0,0]], "SYMMETRIC")
        h_x = (h_padded[:, 2:, :] - h_padded[:, :-2, :]) / (2 * self.dx)
        u_x = (u_padded[:, 2:, :] - u_padded[:, :-2, :]) / (2 * self.dx)
        
        g = 9.81
        S0 = self.params.physics["slope"]
        n_manning = self.params.physics["manning"]
        h_safe = tf.maximum(h, 0.1)
        Sf = (n_manning**2 * u * tf.abs(u)) / (h_safe**(4/3))
        
        res_mass = h_t + h * u_x + u * h_x
        res_mom = u_t_momentum + u * u_x + g * h_x + g * (Sf - S0)
        return res_mass, res_mom

    def compute_gradient_weights(self, target):
        target_padded = tf.pad(target, [[0,0], [1,1], [0,0]], "SYMMETRIC")
        target_x = (target_padded[:, 2:, :] - target_padded[:, :-2, :]) / (2 * self.dx)
        grad_mag = tf.abs(target_x)
        return 1.0 + self.beta_grad * grad_mag

    @tf.function
    def train_step(self, inputs, target, bc_next):
        u_prev_norm = inputs[:, -1, :, 0:2]
        with tf.GradientTape() as tape:
            pred = self.model(inputs)
            pred = tf.cast(pred, tf.float32) 
            
            weights = self.compute_gradient_weights(target)
            weighted_sq_diff = weights * tf.square(pred - target)
            loss_data = tf.reduce_mean(weighted_sq_diff)
            
            res_mass, res_mom = self.compute_pde_residual(pred, u_prev_norm)
            loss_pde = tf.reduce_mean(tf.square(res_mass)) + tf.reduce_mean(tf.square(res_mom))
            
            loss = self.lambda_data * loss_data + self.lambda_pde * loss_pde
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, loss_data, loss_pde

    def train(self):
        print(f"Starting ST-APINO Training for {self.epochs} epochs...")
        
        # Load ALL data into memory (NumPy arrays)
        train_inputs, train_targets, train_bcs = self.dataset.get_data_arrays(mode='train')
        N_train = len(train_inputs)
        steps_per_epoch = N_train // self.batch_size
        
        print(f"Train Samples: {N_train}, Batch Size: {self.batch_size}, Steps: {steps_per_epoch}")
        
        start_time = time.time()
        best_error = float('inf')
        
        for epoch in range(self.epochs):
            total_loss = 0; total_data = 0; total_pde = 0
            
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
                total_data += l_d.numpy()
                total_pde += l_p.numpy()
                
            avg_loss = total_loss / steps_per_epoch
            
            # Logging
            elapsed = time.time() - start_time
            log_str = f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.6f} (D:{total_data/steps_per_epoch:.6f} P:{total_pde/steps_per_epoch:.6f})"
            
            # Validation Every 10 Epochs
            if (epoch + 1) % 10 == 0:
                val_error = validate_model(self.model, self.params, self.dataset, epoch=epoch+1, save_dir=self.plot_dir)
                log_str += f" | Val Err: {val_error:.4%}"
                
                # Save Checkpoint
                self.model.save_weights(os.path.join(self.ckpt_dir, f"model_epoch_{epoch+1}.weights.h5"))
                
                if val_error < best_error:
                    best_error = val_error
                    self.model.save_weights(os.path.join(self.ckpt_dir, "model_best.weights.h5"))
                    log_str += " [BEST]"
            
            print(log_str + f" | Time: {elapsed:.1f}s")