from utility import set_directory, load_json, switch_dataset, switch_equation
from setup import Param, Dataset
from networks import BayesNN
from networks.Theta import Theta
import tensorflow as tf
import numpy as np
import os

# Setup
set_directory()
config_file = "best_models/HMC_sv_1d"
config = load_json(config_file)

class Args:
    def __init__(self):
        self.config = "HMC_sv_1d"
        self.problem = None
        self.case_name = None
        self.method = None
        self.epochs = None
        self.save_flag = False
        self.gen_flag = False
        self.debug_flag = False
        self.random_seed = 42
args = Args()

params = Param(config, args)
data_config = switch_dataset(params.problem, params.case_name)
params.data_config = data_config

# Load Dataset (to get norm coefficients)
dataset = Dataset(params)

# Build Model
equation = switch_equation(params.problem)
bayes_nn = BayesNN(params, equation)

# Set normalization coefficients
bayes_nn.u_coeff = dataset.norm_coeff["sol_mean"], dataset.norm_coeff["sol_std"]
bayes_nn.f_coeff = dataset.norm_coeff["par_mean"], dataset.norm_coeff["par_std"]
bayes_nn.norm_coeff = dataset.norm_coeff

# Load Pre-trained Weights
path_pretrained = "../pretrained_models/pretrained_SaintVenant1D_simple_ADAM.npy"
if not os.path.exists(path_pretrained):
    print("Pre-trained weights not found!")
    exit()

print("Loading weights...")
loaded_values = np.load(path_pretrained, allow_pickle=True)
theta_values = [tf.convert_to_tensor(v, dtype=tf.float32) for v in loaded_values]
bayes_nn.nn_params = Theta(theta_values)

# Generate Test Points
print("Generating test points...")
# Using the same logic as main_plot_only to target a specific location (e.g. x=2km)
L = params.physics["length"]
T = params.physics["time"]
num_t = 100
t_h = np.linspace(0, 4, num_t)
t_norm = t_h / (T / 3600.0)
x_km = 2.0
x_norm_val = (x_km * 1000.0) / L
x_col = np.full(num_t, x_norm_val)
inputs_np = np.stack([x_col, t_norm], axis=1)
inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)

# Compute Residuals Manually
print("Computing residuals and gradients...")
with tf.GradientTape(persistent=True) as tape:
    tape.watch(inputs)
    # Forward pass (returns normalized output)
    out_sol, out_par = bayes_nn.forward(inputs)
    
    # Unpack and Denormalize (Logic from SaintVenant.py)
    # Note: we need to replicate comp_residual logic exactly
    from equations.Operators import Operators
    sol_list = Operators.tf_unpack(out_sol)
    h_norm = sol_list[0]
    u_norm = sol_list[1]
    
    h_mu = bayes_nn.norm_coeff["h_mean"]
    h_sigma = bayes_nn.norm_coeff["h_std"]
    u_mu = bayes_nn.norm_coeff["u_mean"]
    u_sigma = bayes_nn.norm_coeff["u_std"]
    
    h = h_norm * h_sigma + h_mu
    u = u_norm * u_sigma + u_mu
    
    # Constraint (from SaintVenant.py)
    # h = tf.math.softplus(h) + 0.5 # Was removed in latest version? 
    # Let's check if it was removed. Yes, replaced with maximum in friction.
    # But let's match current code state.
    # Current code has NO modification to h here.
    
    # Gradients
    grad_h_norm = Operators.gradient_scalar(tape, h_norm, inputs)
    grad_u_norm = Operators.gradient_scalar(tape, u_norm, inputs)

# Compute Physical Gradients
inv_L = 1.0 / L
inv_T = 1.0 / T

h_x = grad_h_norm[:, 0:1] * h_sigma * inv_L
h_t = grad_h_norm[:, 1:2] * h_sigma * inv_T
u_x = grad_u_norm[:, 0:1] * u_sigma * inv_L
u_t = grad_u_norm[:, 1:2] * u_sigma * inv_T

# Compute Residuals
g = 9.81
slope = 0.001
manning = 0.03

h_safe = tf.maximum(h, 0.5)
Sf = (manning**2 * u * tf.abs(u)) / (h_safe**(4/3))

lhs_cont = h_t + (h_x * u + h * u_x)
lhs_mom = u_t + u * u_x + g * h_x - g * (slope - Sf)

res_cont = tf.reduce_mean(tf.square(lhs_cont))
res_mom = tf.reduce_mean(tf.square(lhs_mom))

print(f"\n--- Statistics at x={x_km}km ---")
print(f"h: Mean={np.mean(h):.4f}, Std={np.std(h):.4f}, Range=[{np.min(h):.4f}, {np.max(h):.4f}]")
print(f"u: Mean={np.mean(u):.4f}, Std={np.std(u):.4f}, Range=[{np.min(u):.4f}, {np.max(u):.4f}]")

print(f"\n--- Gradients (Physical Units) ---")
print(f"h_x: Mean={np.mean(h_x):.2e}, MaxAbs={np.max(np.abs(h_x)):.2e}")
print(f"h_t: Mean={np.mean(h_t):.2e}, MaxAbs={np.max(np.abs(h_t)):.2e}")
print(f"u_x: Mean={np.mean(u_x):.2e}, MaxAbs={np.max(np.abs(u_x)):.2e}")
print(f"u_t: Mean={np.mean(u_t):.2e}, MaxAbs={np.max(np.abs(u_t)):.2e}")

print(f"\n--- Residuals (MSE) ---")
print(f"Continuity: {res_cont:.2e}")
print(f"Momentum:   {res_mom:.2e}")
print(f"Total PDE:  {res_cont + res_mom:.2e}")

# Compare with training log value (approx 3.6e-6)
