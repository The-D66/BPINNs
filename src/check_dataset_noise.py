from utility import set_directory, load_json, switch_dataset
from setup import Param, Dataset
import numpy as np
import tensorflow as tf
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

print("Initializing Dataset (which applies noise)...")
dataset = Dataset(params)

# Check Boundary Noise
print("\n--- Checking Boundary Noise ---")
# data_bnd["sol"] contains [h, u] with noise
bnd_noisy = dataset.data_bnd["sol"]
bnd_coords = dataset.data_bnd["dom"]

# Get Exact values
# data_config.values["u"] expects tuple of coordinates
exact_vals = data_config.values["u"]([bnd_coords[:,0], bnd_coords[:,1]])
# returns list [h, u]
h_exact = exact_vals[0].flatten()
u_exact = exact_vals[1].flatten()

h_noisy = bnd_noisy[:,0]
u_noisy = bnd_noisy[:,1]

diff_h = h_noisy - h_exact
diff_u = u_noisy - u_exact

std_h = np.std(diff_h)
std_u = np.std(diff_u)

print(f"Boundary Data Points: {len(h_noisy)}")
print(f"Measured Noise Std (h): {std_h:.4f} (Target ~0.2)")
print(f"Measured Noise Std (u): {std_u:.4f} (Target ~0.5)")

# Check Internal Noise (sol)
print("\n--- Checking Internal Solution Noise ---")
sol_noisy = dataset.data_sol["sol"]
sol_coords = dataset.data_sol["dom"]

if len(sol_noisy) > 0:
    exact_vals_sol = data_config.values["u"]([sol_coords[:,0], sol_coords[:,1]])
    h_exact_sol = exact_vals_sol[0].flatten()
    u_exact_sol = exact_vals_sol[1].flatten()
    
    diff_h_sol = sol_noisy[:,0] - h_exact_sol
    diff_u_sol = sol_noisy[:,1] - u_exact_sol
    
    print(f"Internal Data Points: {len(sol_noisy)}")
    print(f"Measured Noise Std (h): {np.std(diff_h_sol):.4f}")
    print(f"Measured Noise Std (u): {np.std(diff_u_sol):.4f}")
else:
    print("No internal solution points used.")
