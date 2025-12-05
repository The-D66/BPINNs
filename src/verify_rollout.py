import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Compiled Rollout Step for Speed
@tf.function
def rollout_step(model, history_buffer, mask_repeated, bc_in, bc_out):
    # history_buffer: (1, 120, Nx, 2)
    # mask_repeated: (1, 120, Nx, 1)
    
    # Prepare Input
    model_input = tf.concat([history_buffer, mask_repeated], axis=-1)
    
    # Predict
    pred_norm = model(model_input, training=False) # (1, Nx, 2)
    
    # Boundary Injection (Hard Constraint) using Tensor operations
    # bc_in/out are scalars or (1, 2) vectors?
    # bc_in: [h, u] at x=0. 
    
    # We need to scatter update. 
    # Or simpler: create a mask for boundary and blend.
    # Actually, tensor_scatter_nd_update is efficient enough.
    
    # But for 1D boundary, simpler approach:
    # Construct the corrected prediction
    # Left boundary (idx 0)
    indices = [[0, 0], [0, 199]] # Batch 0, x=0 and x=Nx-1
    # But pred_norm is (1, Nx, 2).
    
    # Let's just do it with numpy outside for simplicity if tf.scatter is complex?
    # NO, mixing numpy breaks tf.function graph.
    # Let's do slicing.
    
    # Split spatial
    # Left (0), Middle (1:-1), Right (-1)
    
    # Wait, we can just return the raw prediction and fix BC in python loop?
    # That negates the benefit of tf.function if we jump back and forth.
    # Ideally we want the whole loop in TF, but that's hard.
    # Compromise: Compile the Model Call only.
    
    return pred_norm

def validate_model(model, params, dataset, epoch=None, save_dir=None):
    """
    Runs rollout validation on the last sample of the dataset.
    Returns: Mean Relative L2 Error.
    """
    # 1. Prepare Test Data (Last Sample)
    test_idx = -1
    U_GT_full = dataset.field_data[test_idx] # (Nt, Nx, 2)
    
    window_size = params.architecture["window_size"]
    rollout_steps = 240 # 4 hours
    Nt, Nx, _ = U_GT_full.shape
    
    if Nt < window_size + rollout_steps:
        rollout_steps = Nt - window_size
    
    # Normalize
    U_GT_norm = (U_GT_full - dataset.mean) / dataset.std
    
    # Initialize Buffer (Numpy)
    history_buffer = U_GT_norm[:window_size] # (120, Nx, 2)
    history_buffer = history_buffer[np.newaxis, ...] # (1, 120, Nx, 2)
    
    # Mask (Tensor)
    mask_bnd = np.zeros((Nx, 1), dtype=np.float32)
    mask_bnd[0] = 1.0
    mask_bnd[-1] = 1.0
    mask_repeated = np.tile(mask_bnd[np.newaxis, np.newaxis, :, :], (1, window_size, 1, 1))
    mask_tensor = tf.convert_to_tensor(mask_repeated, dtype=tf.float32)
    
    predictions = []
    
    # 2. Rollout Loop
    for t in range(rollout_steps):
        global_t = window_size + t
        
        # Convert buffer to tensor
        hist_tensor = tf.convert_to_tensor(history_buffer, dtype=tf.float32)
        
        # Fast Predict (Compiled)
        pred_norm_tf = rollout_step(model, hist_tensor, mask_tensor, None, None)
        pred_norm = pred_norm_tf.numpy()
        
        # Boundary Injection (Numpy is fast enough for 2 points)
        bc_next_norm = U_GT_norm[global_t]
        pred_norm[0, 0, :] = bc_next_norm[0, :]
        pred_norm[0, -1, :] = bc_next_norm[-1, :]
        
        # Store (Denormalized)
        pred_phys = pred_norm * dataset.std + dataset.mean
        predictions.append(pred_phys[0])
        
        # Update Buffer (Rolling)
        history_buffer = np.concatenate([history_buffer[:, 1:, ...], pred_norm[:, np.newaxis, ...]], axis=1)
        
    predictions = np.array(predictions) # (T, Nx, 2)
    ground_truth = U_GT_full[window_size : window_size + rollout_steps]
    
    # 3. Compute Metric (Relative L2 Error)
    error_l2 = np.linalg.norm(predictions - ground_truth, axis=(1, 2))
    norm_gt = np.linalg.norm(ground_truth, axis=(1, 2)) + 1e-6
    mean_rel_error = np.mean(error_l2 / norm_gt)
    
    # 4. Plotting (if save_dir is provided)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Pick mid-point for profile
        mid_t = rollout_steps // 2
        
        # h profile
        axes[0].plot(ground_truth[mid_t, :, 0], 'k-', label='Exact')
        axes[0].plot(predictions[mid_t, :, 0], 'r--', label='Pred')
        axes[0].set_title(f"Water Depth (Rollout Step {mid_t})")
        axes[0].legend()
        
        # Error trend
        axes[1].plot(error_l2 / norm_gt * 100)
        axes[1].set_title(f"Relative Error % (Mean: {mean_rel_error*100:.2f}%)")
        axes[1].set_xlabel("Time Steps")
        
        suffix = f"_epoch_{epoch}" if epoch is not None else ""
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"val_rollout{suffix}.png"))
        plt.close()
        
    return mean_rel_error

if __name__ == "__main__":
    # Standalone run logic
    import json
    from setup.param import Param
    from networks.ConvFormer import ConvFormer
    from setup.window_loader import WindowDataset
    
    class Args:
        def __init__(self):
            self.config = "autoregressive_sv"
            self.problem = None; self.case_name = None; self.method = None
            self.epochs = None; self.save_flag = False; self.gen_flag = False
            self.debug_flag = False; self.random_seed = 42
    
    params = Param(json.load(open("../config/autoregressive_sv.json")), Args())
    dataset = WindowDataset(params)
    model = ConvFormer(params)
    _ = model(tf.zeros((1, 120, 200, 3)))
    
    ckpt = f"../outs/{params.problem}/checkpoints/model_best.weights.h5"
    if os.path.exists(ckpt):
        print(f"Loading best model: {ckpt}")
        model.load_weights(ckpt)
        
    err = validate_model(model, params, dataset, save_dir="../outs/verify_rollout")
    print(f"Validation Error: {err:.4f}")
