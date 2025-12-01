import os
# Environment variables for safety (even without matplotlib, TF can have issues)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import tensorflow as tf
import sys
import numpy as np
import timeit

# Configuration
L_PHYS = 10000.0
T_PHYS = 14400.0
SLOPE = 0.001
MANNING = 0.03
G = 9.81
H0 = 5.0
Q0 = 20.0

EPOCHS = 20000
LEARNING_RATE_1 = 1e-3
LEARNING_RATE_2 = 1e-4
OUT_DIR = "outs/clean_solve"

def load_reference_data():
    path_raw = "data_raw"
    try:
        h_hist = np.load(os.path.join(path_raw, "h_history.npy"))
        u_hist = np.load(os.path.join(path_raw, "u_history.npy"))
        t_grid = np.load(os.path.join(path_raw, "t_grid.npy"))
        x_grid = np.load(os.path.join(path_raw, "x_grid.npy"))
        config = np.load(os.path.join(path_raw, "config.npy"), allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    T_warmup = config.get("T_warmup", 3600.0)
    start_idx = np.searchsorted(t_grid, T_warmup)
    t_grid_crop = t_grid[start_idx:] - T_warmup
    h_hist_crop = h_hist[start_idx:, :]
    u_hist_crop = u_hist[start_idx:, :]
    
    end_idx = np.searchsorted(t_grid_crop, T_PHYS)
    if end_idx < len(t_grid_crop):
        t_grid_crop = t_grid_crop[:end_idx+1]
        h_hist_crop = h_hist_crop[:end_idx+1, :]
        u_hist_crop = u_hist_crop[:end_idx+1, :]
    
    return x_grid, t_grid_crop, h_hist_crop, u_hist_crop

def get_boundary_data(x_grid, t_grid, h_exact, u_exact, num_points_t, num_points_x):
    idx_t0 = 0
    x_ic = x_grid
    h_ic = h_exact[idx_t0, :]
    u_ic = u_exact[idx_t0, :]
    
    idx_ic = np.linspace(0, len(x_ic)-1, num_points_x, dtype=int)
    x_ic_train = x_ic[idx_ic] / L_PHYS
    t_ic_train = np.zeros_like(x_ic_train)
    u_ic_train = u_ic[idx_ic]
    h_ic_train = h_ic[idx_ic]
    
    idx_x0 = 0
    h_bc_l = h_exact[:, idx_x0]
    u_bc_l = u_exact[:, idx_x0]
    
    idx_bcl = np.linspace(0, len(t_grid)-1, num_points_t, dtype=int)
    x_bcl_train = np.zeros_like(idx_bcl, dtype=float)
    t_bcl_train = t_grid[idx_bcl] / T_PHYS
    h_bcl_train = h_bc_l[idx_bcl]
    u_bcl_train = u_bc_l[idx_bcl]
    
    idx_xL = -1
    h_bc_r = h_exact[:, idx_xL]
    u_bc_r = u_exact[:, idx_xL]
    
    idx_bcr = np.linspace(0, len(t_grid)-1, num_points_t, dtype=int)
    x_bcr_train = np.ones_like(idx_bcr, dtype=float)
    t_bcr_train = t_grid[idx_bcr] / T_PHYS
    h_bcr_train = h_bc_r[idx_bcr]
    u_bcr_train = u_bc_r[idx_bcr]
    
    return (x_ic_train, t_ic_train, u_ic_train, h_ic_train), \
           (x_bcl_train, t_bcl_train, u_bcl_train, h_bcl_train), \
           (x_bcr_train, t_bcr_train, u_bcr_train, h_bcr_train)

class PINN(tf.keras.Model):
    def __init__(self, neurons=32, layers=6, activation='tanh'):
        super(PINN, self).__init__()
        self.hidden_layers = []
        for _ in range(layers - 1):
            self.hidden_layers.append(tf.keras.layers.Dense(neurons, activation=activation))
        self.output_layer = tf.keras.layers.Dense(2)

    def call(self, inputs):
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
        h = tf.nn.softplus(h)
    u_x = tape.gradient(u, x)
    u_t = tape.gradient(u, t)
    h_x = tape.gradient(h, x)
    h_t = tape.gradient(h, t)
    del tape
    return u, h, u_x/L_PHYS, u_t/T_PHYS, h_x/L_PHYS, h_t/T_PHYS

def physics_residual(model, x, t):
    u, h, u_x, u_t, h_x, h_t = get_derivatives(model, x, t)
    R = tf.maximum(h, 0.01)
    Sf = (MANNING**2 * u**2) / (R**(4/3))
    res_mass = h_t + h * u_x + u * h_x
    res_mom = u_t + u * u_x + G * h_x + G * (Sf - SLOPE)
    return res_mass, res_mom

def compute_loss(model, x_coll, t_coll, bc_data):
    (x_ic, t_ic, u_ic, h_ic), (x_bcl, t_bcl, u_bcl, h_bcl), (x_bcr, t_bcr, u_bcr, h_bcr) = bc_data
    
    res_mass, res_mom = physics_residual(model, x_coll, t_coll)
    loss_pde = tf.reduce_mean(tf.square(res_mass)) + tf.reduce_mean(tf.square(res_mom))
    
    out_ic = model(tf.concat([x_ic[:,None], t_ic[:,None]], axis=1))
    loss_ic = tf.reduce_mean(tf.square(out_ic[:, 0:1] - u_ic[:,None])) + \
              tf.reduce_mean(tf.square(tf.nn.softplus(out_ic[:, 1:2]) - h_ic[:,None]))
              
    out_bcl = model(tf.concat([x_bcl[:,None], t_bcl[:,None]], axis=1))
    loss_bcl = tf.reduce_mean(tf.square(out_bcl[:, 0:1] - u_bcl[:,None])) + \
               tf.reduce_mean(tf.square(tf.nn.softplus(out_bcl[:, 1:2]) - h_bcl[:,None]))
               
    out_bcr = model(tf.concat([x_bcr[:,None], t_bcr[:,None]], axis=1))
    loss_bcr = tf.reduce_mean(tf.square(tf.nn.softplus(out_bcr[:, 1:2]) - h_bcr[:,None]))
    
    return loss_pde + 10.0*(loss_ic + loss_bcl + loss_bcr)

@tf.function
def train_step(model, optimizer, x_coll, t_coll, bc_data):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x_coll, t_coll, bc_data)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "log"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "plot"), exist_ok=True)
    # Create dummy parameters.txt for Plotter compatibility
    with open(os.path.join(OUT_DIR, "log", "parameters.txt"), "w") as f:
        f.write("Clean Solve Parameters\n")
        f.write("Case: Clean Solve\n")
        f.write("Problem : SaintVenant1D\n") # Triggers scaling in Plotter

    x_grid, t_grid, h_exact, u_exact = load_reference_data()
    
    N_coll = 5000
    x_coll = tf.random.uniform((N_coll, 1), 0, 1, dtype=tf.float32)
    t_coll = tf.random.uniform((N_coll, 1), 0, 1, dtype=tf.float32)
    
    bc_data_np = get_boundary_data(x_grid, t_grid, h_exact, u_exact, 100, 100)
    bc_data = []
    for group in bc_data_np:
        bc_data.append([tf.convert_to_tensor(arr, dtype=tf.float32) for arr in group])
        
    model = PINN()
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[EPOCHS // 2], values=[LEARNING_RATE_1, LEARNING_RATE_2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    print("Starting Training (No Plotting)...")
    history_loss = []
    start_time = timeit.default_timer()
    
    for epoch in range(EPOCHS):
        loss = train_step(model, optimizer, x_coll, t_coll, bc_data)
        history_loss.append(loss.numpy())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")
            
    print(f"Training finished in {timeit.default_timer() - start_time:.2f}s")
    
    # Predictions
    Tv, Xv = np.meshgrid(t_grid, x_grid, indexing='ij')
    X_full = Xv.flatten()[:, None]
    T_full = Tv.flatten()[:, None]
    inputs_full = np.hstack((X_full/L_PHYS, T_full/T_PHYS))
    
    # Chunked prediction
    preds_h, preds_u = [], []
    chunk_size = 5000
    for i in range(0, len(inputs_full), chunk_size):
        batch = tf.convert_to_tensor(inputs_full[i:i+chunk_size], dtype=tf.float32)
        out = model(batch)
        preds_u.append(out[:, 0:1].numpy())
        preds_h.append(tf.nn.softplus(out[:, 1:2]).numpy())
        
    Y_pred = np.hstack((np.vstack(preds_h), np.vstack(preds_u)))
    Y_exact = np.hstack((h_exact.flatten()[:, None], u_exact.flatten()[:, None]))
    
    # Save Data for Plotting
    np.save(os.path.join(OUT_DIR, "history_loss.npy"), np.array(history_loss))
    np.save(os.path.join(OUT_DIR, "inputs_full.npy"), inputs_full)
    np.save(os.path.join(OUT_DIR, "Y_pred.npy"), Y_pred)
    np.save(os.path.join(OUT_DIR, "Y_exact.npy"), Y_exact)
    
    print(f"Data saved to {OUT_DIR}")

if __name__ == "__main__":
    main()