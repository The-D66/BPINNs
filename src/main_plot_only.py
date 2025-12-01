# %% Utilities
from utility import set_config, set_directory, set_warning, starred_print
from utility import load_json, check_dataset, create_directories
from utility import switch_dataset, switch_equation, switch_configuration

# Setup utilities
set_directory()
set_warning()

# %% Import Local Classes

from setup import Parser, Param             # Setup
from setup import DataGenerator, Dataset    # Dataset Creation
from networks import BayesNN                # Models
from networks.Theta import Theta
from algorithms import Trainer              # Algorithms
from postprocessing import Storage, Plotter # Postprocessing
import os
import numpy as np
import tensorflow as tf

# %% Creating Parameters

starred_print("START PLOTTING ONLY")
# Manually set config
configuration_file = "best_models/HMC_sv_1d" 
config = load_json(configuration_file)  
# Create a dummy args object
class Args:
    def __init__(self):
        self.config = "HMC_sv_1d"
        self.problem = None
        self.case_name = None
        self.method = None
        self.epochs = None
        self.save_flag = True
        self.gen_flag = False
        self.debug_flag = False
        self.random_seed = None
args = Args()

params = Param(config, args)     # Combines args and config

data_config = switch_dataset(params.problem, params.case_name)
params.data_config = data_config

print(f"Plotting for {params.problem} - {params.case_name}")

# %% Datasets (Load only)
dataset = Dataset(params)

# %% Directories
# We need to find the latest output folder for HMC
# Use the specific folder requested by user
path_folder = "../outs/SaintVenant1D/SaintVenant1D/HMC_2025.11.25-12.18.52"

if not os.path.exists(path_folder):
    print(f"Path not found: {path_folder}")
    exit()

print("Loading data...")
plotter = Plotter(path_folder)
load_storage = Storage(path_folder)

# Prepare plot data (Exact solution, etc.)
try:
    plot_data = load_storage.data
except Exception as e:
    print(f"Storage data missing ({e}). Using dataset directly.")
    plot_data = dataset.data_plot

# %% Model Building (for Inference using HMC samples)

print("Building Model for Inference...")
equation = switch_equation(params.problem)
bayes_nn = BayesNN(params, equation)

# Set normalization coefficients manually from dataset
bayes_nn.u_coeff = dataset.norm_coeff["sol_mean"], dataset.norm_coeff["sol_std"]
bayes_nn.f_coeff = dataset.norm_coeff["par_mean"], dataset.norm_coeff["par_std"]
bayes_nn.norm_coeff = dataset.norm_coeff

# Load HMC Thetas from Storage
print("Loading HMC samples from storage...")
try:
    # load_storage.thetas returns a list of lists of numpy arrays. 
    # Theta constructor expects a list of tensors.
    loaded_thetas_raw = load_storage.thetas
    
    if not loaded_thetas_raw:
        print("No HMC samples found in storage. Checking for pre-trained weights...")
        # Fallback to pre-trained if HMC list is empty (e.g. if only pre-training ran)
        path_pretrained = f"../pretrained_models/pretrained_{params.problem}_{params.case_name}_ADAM.npy"
        if os.path.exists(path_pretrained):
             loaded_values = np.load(path_pretrained, allow_pickle=True)
             theta_values = [tf.convert_to_tensor(v, dtype=tf.float32) for v in loaded_values]
             bayes_nn.thetas = [Theta(theta_values)]
             print("Loaded pre-trained weights (1 sample).")
        else:
             print("No weights found.")
    else:
        bayes_nn.thetas = []
        for theta_list in loaded_thetas_raw:
            # theta_list is [w1, b1, w2, b2, ...] (numpy arrays)
            theta_values = [tf.convert_to_tensor(v, dtype=tf.float32) for v in theta_list]
            bayes_nn.thetas.append(Theta(theta_values))
        print(f"Loaded {len(bayes_nn.thetas)} HMC samples.")

    # Use the last sample as the 'current' parameter for any single-shot operations if needed
    if bayes_nn.thetas:
        bayes_nn.nn_params = bayes_nn.thetas[-1]

    print("Computing predictions with uncertainty...")
    # Predict on test domain using loaded weights
    # self.data_test["dom"] is normalized [0,1]
    # functions_confidence will contain physical values (h, u)
    functions_confidence = bayes_nn.mean_and_std(dataset.data_test["dom"])
    
    print("Plotting Standard Results...")
    # Use plot_data here
    plotter.plot_confidence(plot_data, functions_confidence)

except Exception as e:
    print(f"Failed to compute predictions from weights: {e}")
    print("Attempting to load stored results directly...")
    # Fallback to stored results if prediction fails
    functions_confidence = load_storage.confidence

# %% Plotting (Original History and Data Dist)

print("Loading stored history...")
try:
    print("Plotting the history...")
    plotter.plot_losses(load_storage.history)
except Exception as e:
    print(f"Failed to plot history: {e}")

# Plot training data distribution
print("Plotting training data distribution (excluding PDE points)...")
# Pass None for pde_data to hide dense PDE points
# Use plot_data for background if needed, but plot_training_data expects `data` struct.
# dataset.data_bnd is correct.
plotter.plot_training_data(plot_data, None, dataset.data_bnd, dataset.data_sol)

# Plot QQ
print("Plotting QQ plots...")
plotter.plot_qq(plot_data, functions_confidence)

# Plot Error Distribution for Training Points
print("Plotting Error Distribution for Training Points (excluding PDE points)...")

# Collect all training data points (coordinates)
all_train_coords_list = []
train_types = []
train_labels = []

if dataset.data_sol["dom"].shape[0] > 0:
    all_train_coords_list.append(dataset.data_sol["dom"])
    train_types.append("sol")
    train_labels.append("Solution Points")

if dataset.data_bnd["dom"].shape[0] > 0:
    all_train_coords_list.append(dataset.data_bnd["dom"])
    train_types.append("bnd")
    train_labels.append("Boundary Points")

# PDE points excluded to reduce clutter
# if dataset.data_pde["dom"].shape[0] > 0:
#     all_train_coords_list.append(dataset.data_pde["dom"])
#     train_types.append("pde")
#     train_labels.append("PDE Points")

if len(all_train_coords_list) > 0:
    all_train_coords = np.concatenate(all_train_coords_list, axis=0)
    
    # Predict NN outputs for these training points
    functions_confidence_train = bayes_nn.mean_and_std(all_train_coords)
    
    # Get Exact values for these training points
    all_train_coords_tf = tf.constant(all_train_coords.T, dtype=tf.float32)
    h_ex_train, u_ex_train = data_config.values["u"](all_train_coords_tf)
    
    # Combine everything for plot_error_distribution
    # Recreate the data structure expected by plot_error_distribution
    
    print(f"Shape of functions_confidence_train['sol_NN']: {functions_confidence_train['sol_NN'].shape}")
    
    train_data_for_plot = {
        "coords": all_train_coords,
        "h_ex": h_ex_train,
        "u_ex": u_ex_train,
        "h_nn": functions_confidence_train['sol_NN'][:,0],
        "u_nn": functions_confidence_train['sol_NN'][:,1],
        "types": train_types, # Store types to identify points in plot if needed
        "labels": train_labels
    }
    
    plotter.plot_error_distribution(train_data_for_plot)
else:
    print("No training data points found to plot error distribution.")

# Plot Full Domain Error Distribution
print("Plotting Full Domain Error Distribution...")
plotter.plot_full_domain_error(plot_data, functions_confidence)

# --- Plot Time Series at Specific Locations ---
print("Plotting Time Series for h and Q...")
target_locs_km = [0, 2, 4, 6, 8, 10]
loc_data_list = []

# Physical scales
L = params.physics["length"]
T = params.physics["time"]

# Generate time points (0 to T hours, e.g., 100 points)
num_t = 100
t_h_max = params.physics["time"] / 3600.0
t_h = np.linspace(0, t_h_max, num_t) 
t_norm = t_h / t_h_max # Normalized time [0, 1]

for x_km in target_locs_km:
    # Normalized x
    x_norm_val = (x_km * 1000.0) / L
    
    # Create query points (N, 2)
    # x is constant, t varies
    x_col = np.full(num_t, x_norm_val)
    query_points = np.stack([x_col, t_norm], axis=1) # (100, 2)
    
    # Predict NN
    # mean_and_std expects (N, 2)
    preds = bayes_nn.mean_and_std(query_points)
    
    h_mean = preds["sol_NN"][:, 0]
    u_mean = preds["sol_NN"][:, 1]
    h_std  = preds["sol_std"][:, 0]
    u_std  = preds["sol_std"][:, 1]
    
    # Exact values
    # data_config.values["u"] expects tuple/list of tensors/arrays: [x, t] (each (N, 1) or (N,))
    # or (N, 2)? Let's check `boundary_values` in config.
    # It expects input `inputs`. `exact_func` unpacks `inputs[0]` and `inputs[1]`.
    # So we pass `[x_col, t_norm]`.
    exact_res = data_config.values["u"]([x_col, t_norm])
    h_ex = exact_res[0].flatten()
    u_ex = exact_res[1].flatten()
    
    # Calculate Q (Discharge)
    # Exact
    Q_ex = h_ex * u_ex
    
    # NN Mean Q approx
    Q_mean = h_mean * u_mean
    
    # NN Q Std (Error propagation approximation)
    # sigma_Q ~= sqrt( (u * sigma_h)^2 + (h * sigma_u)^2 )
    Q_std = np.sqrt( (u_mean * h_std)**2 + (h_mean * u_std)**2 )
    
    # Find nearby training points for visualization
    # Tolerance for 'nearby': say 1% of domain length (100m)
    x_tol = 0.01 
    
    train_pts_t = []
    train_pts_h = []
    train_pts_Q = []
    
    # Search in internal solution points
    if dataset.data_sol["dom"].shape[0] > 0:
        sol_dom = dataset.data_sol["dom"]
        sol_val = dataset.data_sol["sol"]
        mask = np.abs(sol_dom[:,0] - x_norm_val) < x_tol
        if np.any(mask):
            t_found = sol_dom[mask, 1] * (params.physics["time"] / 3600.0)
            h_found = sol_val[mask, 0]
            u_found = sol_val[mask, 1]
            train_pts_t.extend(t_found)
            train_pts_h.extend(h_found)
            train_pts_Q.extend(h_found * u_found)

    # Search in boundary points
    if dataset.data_bnd["dom"].shape[0] > 0:
        bnd_dom = dataset.data_bnd["dom"]
        bnd_val = dataset.data_bnd["sol"]
        mask = np.abs(bnd_dom[:,0] - x_norm_val) < x_tol
        if np.any(mask):
            t_found = bnd_dom[mask, 1] * (params.physics["time"] / 3600.0)
            h_found = bnd_val[mask, 0]
            u_found = bnd_val[mask, 1]
            train_pts_t.extend(t_found)
            train_pts_h.extend(h_found)
            train_pts_Q.extend(h_found * u_found)
            
    loc_data = {
        "x_km": x_km,
        "t_h": t_h,
        "h_ex": h_ex,
        "h_mean": h_mean,
        "h_std": h_std,
        "Q_ex": Q_ex,
        "Q_mean": Q_mean,
        "Q_std": Q_std,
        "train_t": np.array(train_pts_t),
        "train_h": np.array(train_pts_h),
        "train_Q": np.array(train_pts_Q)
    }
    loc_data_list.append(loc_data)

plotter.plot_time_series(loc_data_list)

starred_print("END")

plotter.show_plot()
