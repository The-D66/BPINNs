
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Add src to path for importing Plotter
sys.path.append(os.path.join(os.getcwd(), 'src'))
from postprocessing.Plotter import Plotter

# Configuration
EPOCHS = 500 # Reduced for demonstration speed
NOISE_H = 0.2
NOISE_U = 0.5
SLOPE = 0.001
MANNING = 0.03
LENGTH = 10000.0
TIME = 14400.0
OUT_DIR = "outs/noisy_solve"

# Ensure output directories exist
os.makedirs(os.path.join(OUT_DIR, "log"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plot"), exist_ok=True)

# Create dummy parameters.txt for Plotter
with open(os.path.join(OUT_DIR, "log", "parameters.txt"), "w") as f:
    f.write("Dummy Parameters File\n")
    f.write("Case: Noisy Solve\n")
    f.write("Problem : SaintVenant1D\n")

# ==========================================
# 1. Data Loading and Noise Generation
# ==========================================
def load_and_noise_data():
    path_raw = "data_raw"
    print(f"Loading data from {path_raw}...")
    
    h_history = np.load(os.path.join(path_raw, "h_history.npy"))
    u_history = np.load(os.path.join(path_raw, "u_history.npy"))
    t_grid = np.load(os.path.join(path_raw, "t_grid.npy"))
    x_grid = np.load(os.path.join(path_raw, "x_grid.npy"))
    
    # Apply Noise
    np.random.seed(42)
    noise_h = np.random.normal(0, NOISE_H, h_history.shape)
    noise_u = np.random.normal(0, NOISE_U, u_history.shape)
    
    h_noisy = h_history + noise_h
    u_noisy = u_history + noise_u
    
    # Prepare Training Data (Flattened)
    # We use the full domain as "noisy observations"
    Tv, Xv = np.meshgrid(t_grid, x_grid, indexing='ij')
    
    # Normalize Inputs [0, 1]
    t_flat = Tv.flatten()[:, None] / TIME
    x_flat = Xv.flatten()[:, None] / LENGTH
    
    h_flat = h_noisy.flatten()[:, None]
    u_flat = u_noisy.flatten()[:, None]
    
    h_ex_flat = h_history.flatten()[:, None]
    u_ex_flat = u_history.flatten()[:, None]
    
    # Stack inputs: (t, x) -> (N, 2) 
    # Note: Plotter expects (x, t) order for coords usually, checking StVenant-PINN uses (x, t) as inputs.
    # But PINN model usually takes (x, t) or stacked.
    # Let's use (x, t) order for model inputs to match typical PINN convention.
    
    X_train = np.hstack((x_flat, t_flat)) # (N, 2) -> x, t
    Y_train = np.hstack((u_flat, h_flat)) # (N, 2) -> u, h (Order matters for PINN output)
    Y_exact = np.hstack((u_ex_flat, h_ex_flat))

    return X_train, Y_train, Y_exact, t_grid, x_grid

X_train_full, Y_train_full, Y_exact_full, t_grid_raw, x_grid_raw = load_and_noise_data()

# Subsample for training (don't use all pixels, too slow)
N_train = 5000
idx = np.random.choice(X_train_full.shape[0], N_train, replace=False)
X_train_batch = tf.convert_to_tensor(X_train_full[idx], dtype=tf.float32)
Y_train_batch = tf.convert_to_tensor(Y_train_full[idx], dtype=tf.float32) # u, h

# Collocation points for PDE (random in domain)
N_pde = 5000
X_pde = tf.random.uniform((N_pde, 2), minval=0.0, maxval=1.0, dtype=tf.float32)

# ==========================================
# 2. PINN Model (from demo/StVenant-PINN.py)
# ==========================================
class PINN(tf.keras.Model):
    def __init__(self, neurons=32, layers=5, activation='tanh'):
        super(PINN, self).__init__()
        self.hidden_layers = []
        for _ in range(layers - 1):
            self.hidden_layers.append(tf.keras.layers.Dense(neurons, activation=activation))
        self.output_layer = tf.keras.layers.Dense(2) # u, h

    def call(self, inputs):
        z = inputs
        for layer in self.hidden_layers:
            z = layer(z)
        return self.output_layer(z)

# Physics Functions (Adapted for StVenant1D_simple)
# x and t are normalized [0,1] inside the network, but physics equations need physical units?
# Or we chain rule the derivatives.
# StVenant equations:
# h_t + (hu)_x = 0
# u_t + u u_x + g h_x + g(Sf - S0) = 0
# Derivatives: ∂h/∂t = (∂h/∂T) * (∂T/∂t) = (∂h/∂T) * (1/TIME)
# ∂h/∂x = (∂h/∂X) * (∂X/∂x) = (∂h/∂X) * (1/LENGTH)

g = 9.81

def get_derivatives(model, xt):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xt)
        output = model(xt)
        u = output[:, 0:1]
        h = output[:, 1:2]
        
        # Softplus for h to ensure positivity (optional but good)
        h = tf.nn.softplus(h) 
        
    # First derivatives
    du_dxt = tape.gradient(u, xt)
    dh_dxt = tape.gradient(h, xt)
    
    u_x = du_dxt[:, 0:1] * (1.0 / LENGTH)
    u_t = du_dxt[:, 1:2] * (1.0 / TIME)
    
    h_x = dh_dxt[:, 0:1] * (1.0 / LENGTH)
    h_t = dh_dxt[:, 1:2] * (1.0 / TIME)
    
    del tape
    return u, h, u_x, u_t, h_x, h_t

def physics_residual(model, xt):
    u, h, u_x, u_t, h_x, h_t = get_derivatives(model, xt)
    
    # Physics Parameters
    # Slope S0 is constant 0.001
    S0 = SLOPE
    
    # Friction Sf = n^2 u^2 / R^(4/3)
    # Wide channel approximation R approx h
    # Avoid division by zero with small epsilon
    n = MANNING
    R = tf.maximum(h, 0.01) 
    Sf = (n**2 * u**2) / (R**(4/3))
    
    # Continuity: h_t + h u_x + u h_x = 0
    res_mass = h_t + h * u_x + u * h_x
    
    # Momentum: u_t + u u_x + g h_x + g(Sf - S0) = 0
    res_mom = u_t + u * u_x + g * h_x + g * (Sf - S0)
    
    return res_mass, res_mom

# Loss Function
def compute_loss(model, xt_data, yh_data, xt_pde):
    # 1. Data Loss (Match noisy observations)
    pred_data = model(xt_data)
    pred_u = pred_data[:, 0:1]
    pred_h = tf.nn.softplus(pred_data[:, 1:2]) # Enforce positive h
    
    true_u = yh_data[:, 0:1]
    true_h = yh_data[:, 1:2]
    
    loss_data = tf.reduce_mean(tf.square(pred_u - true_u)) + \
                tf.reduce_mean(tf.square(pred_h - true_h))
    
    # 2. PDE Loss
    res_mass, res_mom = physics_residual(model, xt_pde)
    loss_pde = tf.reduce_mean(tf.square(res_mass)) + \
               tf.reduce_mean(tf.square(res_mom))
               
    # Weighting
    return loss_data + loss_pde

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(model, xt_data, yh_data, xt_pde):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, xt_data, yh_data, xt_pde)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ==========================================
# 3. Training Loop
# ==========================================
model = PINN()
print("Starting Training...")
history = []

for epoch in range(EPOCHS):
    loss = train_step(model, X_train_batch, Y_train_batch, X_pde)
    history.append(loss.numpy())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")

print("Training Complete.")

# ==========================================
# 4. Post-Processing with Plotter
# ==========================================
print("Preparing plots...")

# Evaluate on full grid for plotting
# X_train_full contains all grid points (N_total, 2)
# Predict in chunks to avoid OOM
pred_u_list = []
pred_h_list = []
chunk_size = 10000
for i in range(0, len(X_train_full), chunk_size):
    batch = X_train_full[i:i+chunk_size]
    out = model(batch)
    pred_u_list.append(out[:, 0:1])
    pred_h_list.append(tf.nn.softplus(out[:, 1:2]))

pred_u = np.vstack(pred_u_list)
pred_h = np.vstack(pred_h_list)
pred_full = np.hstack((pred_h, pred_u)) # h, u order for Plotter?

# Check Plotter expectation:
# __plot_confidence_2D expects func_ex to be (N, dim).
# If dim=2, it plots component 0 and 1.
# Component names in Plotter are ["h", "u"] if dim=2.
# So we should provide [h, u] order.

# Data dictionary for Plotter
# data["sol_ex"] = (coords, values_exact)
# functions["sol_NN"] = values_predicted
# functions["sol_std"] = uncertainty (zeros)

# Plotter expects coords to be (x, t)??
# In __plot_confidence_2D_scalar:
# xx = np.unique(x[:,0]) -> scaled by self.scale_x
# yy = np.unique(x[:,1]) -> scaled by self.scale_t
# So col 0 is x, col 1 is t.
# My X_train_full is (x, t). Perfect.

data_plot = {
    "sol_ex": (X_train_full, Y_exact_full[:, [1, 0]]), # Exact is u, h in my loader?
    # My loader: Y_exact = [u_ex_flat, h_ex_flat].
    # So col 0 is u, col 1 is h.
    # Plotter expects [h, u] order for labels "h", "u".
    # So I need to swap Y_exact cols: [h, u].
    
    "par_ex": (X_train_full, np.zeros_like(X_train_full)), # Dummy
    "sol_ns": (X_train_full[idx], Y_train_full[idx][:, [1, 0]]), # Noisy samples (swapped to h, u)
    "par_ns": None
}

# Prepare predictions [h, u]
pred_full_swapped = np.hstack((pred_h, pred_u))

functions_plot = {
    "sol_NN": pred_full_swapped,
    "sol_std": np.zeros_like(pred_full_swapped), # Deterministic PINN
    "par_NN": np.zeros_like(pred_full_swapped),
    "par_std": np.zeros_like(pred_full_swapped)
}

# Instantiate Plotter
plotter = Plotter(OUT_DIR)
# Plotter scales are set in __init__ based on problem name "SaintVenant"
# scale_x = 10.0, scale_t = 4.0.
# My inputs are normalized.
# X_train_full col 0 is x/L, col 1 is t/T.
# If Plotter multiplies by scale_x=10, it expects inputs in [0,1] representing 10km?
# SaintVenant1D_simple length is 10000m = 10km.
# So yes, normalized inputs are correct.

# Call plotting functions
plotter.plot_confidence(data_plot, functions_plot)
plotter.plot_losses([{"Total": history}, {"Total": history}]) # Hack for loss plot format
plotter.plot_full_domain_error(data_plot, functions_plot)
plotter.show_plot()

print("All Done.")
