
import os
# CRITICAL: Fix for macOS OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt
import timeit

# Add src to path for importing Plotter
sys.path.append(os.path.join(os.getcwd(), 'src'))
from postprocessing.Plotter import Plotter

# ==========================================
# Configuration & Physics
# ==========================================
# Physics Constants (from SaintVenant1D_simple)
L_PHYS = 10000.0
T_PHYS = 14400.0
SLOPE = 0.001
MANNING = 0.03
G = 9.81
H0 = 5.0
Q0 = 20.0

# Training Config
EPOCHS = 2000 # Moderate epochs for demonstration
LEARNING_RATE_1 = 1e-3
LEARNING_RATE_2 = 1e-4
SEGMENT = EPOCHS // 2

# Output
OUT_DIR = "outs/clean_solve"
os.makedirs(os.path.join(OUT_DIR, "log"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plot"), exist_ok=True)

# Create dummy parameters.txt for Plotter compatibility
with open(os.path.join(OUT_DIR, "log", "parameters.txt"), "w") as f:
    f.write("Clean Solve Parameters\n")
    f.write("Case: Clean Solve\n")
    f.write("Problem : SaintVenant1D\n") # Triggers scaling in Plotter

# ==========================================
# Data Preparation
# ==========================================
def load_reference_data():
    path_raw = "data_raw"
    print(f"Loading reference data from {path_raw}...")
    
    try:
        h_hist = np.load(os.path.join(path_raw, "h_history.npy"))
        u_hist = np.load(os.path.join(path_raw, "u_history.npy"))
        t_grid = np.load(os.path.join(path_raw, "t_grid.npy"))
        x_grid = np.load(os.path.join(path_raw, "x_grid.npy"))
        config = np.load(os.path.join(path_raw, "config.npy"), allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Crop Warmup
    T_warmup = config.get("T_warmup", 3600.0)
    start_idx = np.searchsorted(t_grid, T_warmup)
    
    t_grid_crop = t_grid[start_idx:] - T_warmup
    h_hist_crop = h_hist[start_idx:, :]
    u_hist_crop = u_hist[start_idx:, :]
    
    # Verify shapes match T_PHYS
    # Depending on simulation length, crop might be longer or shorter. 
    # We take up to T_PHYS.
    end_idx = np.searchsorted(t_grid_crop, T_PHYS)
    if end_idx < len(t_grid_crop):
        t_grid_crop = t_grid_crop[:end_idx+1]
        h_hist_crop = h_hist_crop[:end_idx+1, :]
        u_hist_crop = u_hist_crop[:end_idx+1, :]
    
    print(f"Data loaded. Shape: {h_hist_crop.shape}, Time: {t_grid_crop[-1]:.1f}s")
    return x_grid, t_grid_crop, h_hist_crop, u_hist_crop

x_grid, t_grid, h_exact, u_exact = load_reference_data()

# Create Meshgrid for full domain evaluation later
Tv, Xv = np.meshgrid(t_grid, x_grid, indexing='ij')
X_full = Xv.flatten()[:, None]
T_full = Tv.flatten()[:, None]
H_full = h_exact.flatten()[:, None]
U_full = u_exact.flatten()[:, None]

# Extract BCs/ICs for Training
# Normalization: Inputs to NN will be [0, 1]
# x_norm = x / L_PHYS
# t_norm = t / T_PHYS

def get_boundary_data(num_points_t, num_points_x):
    # 1. IC: t = 0
    idx_t0 = 0
    x_ic = x_grid
    h_ic = h_exact[idx_t0, :]
    u_ic = u_exact[idx_t0, :]
    
    # Sample IC
    idx_ic = np.linspace(0, len(x_ic)-1, num_points_x, dtype=int)
    x_ic_train = x_ic[idx_ic] / L_PHYS
    t_ic_train = np.zeros_like(x_ic_train)
    u_ic_train = u_ic[idx_ic]
    h_ic_train = h_ic[idx_ic]
    
    # 2. BC Left: x = 0
    idx_x0 = 0
    t_bc = t_grid
    h_bc_l = h_exact[:, idx_x0]
    u_bc_l = u_exact[:, idx_x0]
    
    # Sample BC Left
    idx_bcl = np.linspace(0, len(t_bc)-1, num_points_t, dtype=int)
    x_bcl_train = np.zeros_like(idx_bcl, dtype=float) # 0.0
    t_bcl_train = t_bc[idx_bcl] / T_PHYS
    h_bcl_train = h_bc_l[idx_bcl]
    u_bcl_train = u_bc_l[idx_bcl]
    
    # 3. BC Right: x = L
    idx_xL = -1
    h_bc_r = h_exact[:, idx_xL]
    u_bc_r = u_exact[:, idx_xL] # Not strictly enforced usually, but we can
    
    idx_bcr = np.linspace(0, len(t_bc)-1, num_points_t, dtype=int)
    x_bcr_train = np.ones_like(idx_bcr, dtype=float) # 1.0
    t_bcr_train = t_bc[idx_bcr] / T_PHYS
    h_bcr_train = h_bc_r[idx_bcr]
    u_bcr_train = u_bc_r[idx_bcr]
    
    return (x_ic_train, t_ic_train, u_ic_train, h_ic_train), \
           (x_bcl_train, t_bcl_train, u_bcl_train, h_bcl_train), \
           (x_bcr_train, t_bcr_train, u_bcr_train, h_bcr_train)

# ==========================================
# PINN Model
# ==========================================
class PINN(tf.keras.Model):
    def __init__(self, neurons=32, layers=6, activation='tanh'):
        super(PINN, self).__init__()
        self.hidden_layers = []
        for _ in range(layers - 1):
            self.hidden_layers.append(tf.keras.layers.Dense(neurons, activation=activation))
        self.output_layer = tf.keras.layers.Dense(2) # u, h

    def call(self, inputs):
        # Input: [x, t]
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        return self.output_layer(z)

def get_derivatives(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        xt = tf.concat([x, t], axis=1)
        output = model(xt)
        u = output[:, 0:1]
        h = output[:, 1:2]
        
        # Softplus for h (water depth must be positive)
        h = tf.nn.softplus(h)
        
    u_x = tape.gradient(u, x)
    u_t = tape.gradient(u, t)
    h_x = tape.gradient(h, x)
    h_t = tape.gradient(h, t)
    
    del tape
    
    # Chain rule for normalization
    # x_phys = x_norm * L
    # d/dx_phys = (1/L) * d/dx_norm
    u_x = u_x / L_PHYS
    u_t = u_t / T_PHYS
    h_x = h_x / L_PHYS
    h_t = h_t / T_PHYS
    
    return u, h, u_x, u_t, h_x, h_t

def physics_residual(model, x, t):
    u, h, u_x, u_t, h_x, h_t = get_derivatives(model, x, t)
    
    # Parameters
    S0 = SLOPE
    n = MANNING
    
    # Friction Slope Sf
    R = tf.maximum(h, 0.01)
    Sf = (n**2 * u**2) / (R**(4/3))
    
    # 1. Continuity: h_t + h u_x + u h_x = 0
    # Alternatively: h_t + (hu)_x = 0 -> h_t + h u_x + u h_x
    res_mass = h_t + h * u_x + u * h_x
    
    # 2. Momentum: u_t + u u_x + g h_x + g(Sf - S0) = 0
    res_mom = u_t + u * u_x + G * h_x + G * (Sf - S0)
    
    return res_mass, res_mom

def compute_loss(model, x_coll, t_coll, bc_data):
    # Unpack BC data
    (x_ic, t_ic, u_ic, h_ic), (x_bcl, t_bcl, u_bcl, h_bcl), (x_bcr, t_bcr, u_bcr, h_bcr) = bc_data
    
    # 1. Physics Loss (Collocation Points)
    res_mass, res_mom = physics_residual(model, x_coll, t_coll)
    loss_pde = tf.reduce_mean(tf.square(res_mass)) + tf.reduce_mean(tf.square(res_mom))
    
    # 2. IC Loss
    out_ic = model(tf.concat([x_ic[:,None], t_ic[:,None]], axis=1))
    u_pred_ic = out_ic[:, 0:1]
    h_pred_ic = tf.nn.softplus(out_ic[:, 1:2])
    loss_ic = tf.reduce_mean(tf.square(u_pred_ic - u_ic[:,None])) + \
              tf.reduce_mean(tf.square(h_pred_ic - h_ic[:,None]))
              
    # 3. BC Left Loss
    out_bcl = model(tf.concat([x_bcl[:,None], t_bcl[:,None]], axis=1))
    u_pred_bcl = out_bcl[:, 0:1]
    h_pred_bcl = tf.nn.softplus(out_bcl[:, 1:2])
    loss_bcl = tf.reduce_mean(tf.square(u_pred_bcl - u_bcl[:,None])) + \
               tf.reduce_mean(tf.square(h_pred_bcl - h_bcl[:,None]))
               
    # 4. BC Right Loss (Only h is usually strictly enforced, u is transmissive)
    # But since we have exact data, enforcing both helps convergence.
    out_bcr = model(tf.concat([x_bcr[:,None], t_bcr[:,None]], axis=1))
    h_pred_bcr = tf.nn.softplus(out_bcr[:, 1:2])
    loss_bcr = tf.reduce_mean(tf.square(h_pred_bcr - h_bcr[:,None]))
    
    return loss_pde + 10.0*(loss_ic + loss_bcl + loss_bcr) # Weight BCs higher

@tf.function
def train_step(model, optimizer, x_coll, t_coll, bc_data):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x_coll, t_coll, bc_data)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ==========================================
# Main Execution
# ==========================================
def main():
    # Prepare Training Data
    N_coll = 5000
    N_ic = 100
    N_bc = 100
    
    # Collocation Points (Random in Domain)
    x_coll = tf.random.uniform((N_coll, 1), 0, 1, dtype=tf.float32)
    t_coll = tf.random.uniform((N_coll, 1), 0, 1, dtype=tf.float32)
    
    # BC Data
    bc_data_numpy = get_boundary_data(N_bc, N_ic)
    # Convert to Tensors
    bc_data = []
    for group in bc_data_numpy:
        bc_data.append([tf.convert_to_tensor(arr, dtype=tf.float32) for arr in group])
        
    # Model & Optimizer
    model = PINN()
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[EPOCHS // 2],
        values=[LEARNING_RATE_1, LEARNING_RATE_2]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    print("Starting Training...")
    start_time = timeit.default_timer()
    history_loss = []
    
    for epoch in range(EPOCHS):
        loss = train_step(model, optimizer, x_coll, t_coll, bc_data)
        history_loss.append(loss.numpy())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")
            
    print(f"Training finished in {timeit.default_timer() - start_time:.2f}s")
    
    # Evaluation & Plotting
    print("Evaluating on full domain...")
    
    # Predict in chunks
    chunk_size = 5000
    preds_h = []
    preds_u = []
    
    # Normalize Full Domain inputs
    X_full_norm = X_full / L_PHYS
    T_full_norm = T_full / T_PHYS
    
    inputs_full = np.hstack((X_full_norm, T_full_norm))
    
    for i in range(0, len(inputs_full), chunk_size):
        batch = tf.convert_to_tensor(inputs_full[i:i+chunk_size], dtype=tf.float32)
        out = model(batch)
        preds_u.append(out[:, 0:1].numpy())
        preds_h.append(tf.nn.softplus(out[:, 1:2]).numpy())
        
    pred_h_arr = np.vstack(preds_h)
    pred_u_arr = np.vstack(preds_u)
    
    # Prepare Data for Plotter
    # Plotter expects inputs as [x, t] in physical units?
    # No, Plotter usually expects whatever units scale_x/scale_t are designed for.
    # In Plotter.py: xx = xx * self.scale_x.
    # SaintVenant config in Plotter sets scale_x=10.0 (km), scale_t=4.0 (h).
    # If we pass Normalized inputs [0,1], then 1.0 * 10.0 = 10km. Correct.
    # So we pass Normalized inputs to Plotter.
    
    # Y_exact: [h, u] (Plotter expects this order for labels)
    Y_exact = np.hstack((H_full, U_full))
    Y_pred = np.hstack((pred_h_arr, pred_u_arr))
    
    data_plot = {
        "sol_ex": (inputs_full, Y_exact),
        "par_ex": (inputs_full, np.zeros_like(inputs_full)),
        "sol_ns": (inputs_full[::100], Y_exact[::100]), # Sampled "Exact" points as training dots
        "par_ns": None
    }
    
    functions_plot = {
        "sol_NN": Y_pred,
        "sol_std": np.zeros_like(Y_pred),
        "par_NN": np.zeros_like(Y_pred),
        "par_std": np.zeros_like(Y_pred)
    }
    
    # Plot
    plotter = Plotter(OUT_DIR)
    plotter.plot_confidence(data_plot, functions_plot)
    plotter.plot_losses([{"Total": history_loss}, {"Total": history_loss}])
    plotter.plot_full_domain_error(data_plot, functions_plot)
    
    # Custom QQ Plot call if available or just show
    try:
        plotter.plot_qq(data_plot, functions_plot)
    except:
        pass
        
    plotter.show_plot()

if __name__ == "__main__":
    main()
