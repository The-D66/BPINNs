# %% Utilities
from utility import set_config, set_directory, set_warning, starred_print
from utility import load_json, check_dataset, create_directories
from utility import switch_dataset, switch_equation, switch_configuration
import tensorflow as tf

# Disable GPU to avoid Metal crashes
try:
    tf.config.set_visible_devices([], 'GPU')
    print("GPU disabled for stability.")
except:
    pass

# Setup utilities
set_directory()
set_warning()

# %% Import Local Classes

from setup import Parser, Param             # Setup
from setup import DataGenerator, Dataset    # Dataset Creation
from networks import BayesNN                # Models
from algorithms import Trainer              # Algorithms
from postprocessing import Storage, Plotter # Postprocessing

# %% Creating Parameters

starred_print("START")
configuration_file = switch_configuration("HMC_lap_sin") # Select the configuration file
args   = Parser().parse_args()   # Load a param object from command-line
config_file = set_config(args.config, configuration_file)
config = load_json(config_file)  # Load params from config file
params = Param(config, args)     # Combines args and config

data_config = switch_dataset(params.problem, params.case_name)
params.data_config = data_config

print(f"Bayesian PINN using {params.method}")
print(f"Solve the {params.inverse} problem of {params.pde} {params.phys_dim.n_input}D ")
starred_print("DONE")

# %% Datasets Creation

print("Dataset Creation")
if params.utils["gen_flag"]:
    print("\tGenerating new dataset...")
    DataGenerator(data_config) 
else:
    check_dataset(data_config)
    print(f"\tStored dataset used: {data_config.name}")

dataset = Dataset(params)
starred_print("DONE")

# %% Model Building

print("Building the Model")
print(f"\tChosing {params.pde} equation...")
equation = switch_equation(params.problem)
print("\tInitializing the Bayesian PINN...")
bayes_nn = BayesNN(params, equation) # Initialize the Bayesian NN
starred_print("DONE")

# %% Building saving directories and Plotter
# We need Plotter during training for real-time monitoring
print("Building saving directories...")
path_folder  = create_directories(params)
save_storage = Storage(path_folder)

# Save parameters immediately so Plotter can read them
save_storage.save_parameter(params)

plotter = Plotter(path_folder)

# %% Model Training

print(f"Building all algorithms...")
# Pass plotter and storage to Trainer for real-time plotting
train_algorithm = Trainer(bayes_nn, params, dataset, plotter, save_storage)
train_algorithm.pre_train()
starred_print("DONE")
train_algorithm.train()
starred_print("DONE")

# %% Model Evaluation

test_data = dataset.data_test
dom_eval = test_data["dom"]

# Check for Operator Mode and Resolve Inputs
if getattr(dataset, "operator_mode", False):
    inputs_eval = dataset.resolve_operator_inputs(dom_eval)
else:
    inputs_eval = dom_eval

print("Computing solutions...")
functions_confidence = bayes_nn.mean_and_std(inputs_eval)
functions_nn_samples = bayes_nn.draw_samples(inputs_eval)
print("Computing errors...")
errors = bayes_nn.test_errors(functions_confidence, test_data)
print("Showing errors...")
bayes_nn.show_errors(errors)
starred_print("DONE")

# %% Saving

print("Saving data...")
# Saving Details and Results
save_storage.save_parameter(params)
save_storage.save_errors(errors)
# Saving Dataset
save_storage.data = dataset.data_plot
# Saving Training
save_storage.history  = bayes_nn.history
save_storage.thetas   = bayes_nn.thetas
# Saving Predictions
save_storage.confidence = functions_confidence
save_storage.nn_samples = functions_nn_samples
starred_print("DONE")

# %% Plotting

print("Loading data...")
# Plotter and Storage already initialized
load_storage = Storage(path_folder) # Re-load to ensure consistency if needed, or just use save_storage
print("Plotting the history...")
plotter.plot_losses(load_storage.history)
print("Plotting the results...")
plotter.plot_confidence(load_storage.data, load_storage.confidence)
# Add extra plots for final result
# Note: training_data_distribution and error_distribution might require extra data handling not present in load_storage by default
# But we can try plotting them if main_plot_only logic is integrated or we rely on real-time plots.
# For now, keep standard plots.
# plotter.plot_nn_samples(load_storage.data, load_storage.nn_samples)
starred_print("END")

plotter.show_plot()
