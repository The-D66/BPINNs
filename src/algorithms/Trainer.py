from .HMC import HMC
from .VI import VI
from .SVGD import SVGD
from .ADAM import ADAM
import numpy as np
import os


class Trainer():
  def __init__(self, bayes_nn, params, dataset, plotter=None, storage=None):
    self.debug_flag = params.utils["debug_flag"]
    self.model = bayes_nn
    self.params = params
    self.dataset = dataset
    self.plotter = plotter
    self.storage = storage

  def __switch_algorithm(self, method):
    """ Returns an instance of the class corresponding to the selected method """
    match method:
      case "ADAM":
        return ADAM
      case "HMC":
        return HMC
      case "SVGD":
        return SVGD
      case "VI":
        return VI
      case _:
        raise Exception("This algorithm does not exist!")

  def __algorithm(self, method, par_method):
    algorithm = self.__switch_algorithm(method)
    algorithm = algorithm(self.model, par_method, self.debug_flag)
    algorithm.data_train = self.dataset
    # Inject utilities for real-time plotting
    algorithm.plotter = self.plotter
    algorithm.storage = self.storage
    # Backup physical test data for plotting (before normalization in train)
    algorithm.test_data_phys = self.dataset.data_plot
    # Inject data config for exact solution calculation
    algorithm.data_config = self.params.data_config
    return algorithm

  def pre_train(self):
    if self.params.init is None:
      return

    # Update active losses if override exists in init params
    if "losses" in self.params.param_init:
      print(f"Overriding losses for pre-training: {self.params.param_init['losses']}")
      self.model.update_active_losses(self.params.param_init["losses"])

    # Check for cached pre-trained weights
    import os
    import tensorflow as tf
    import numpy as np
    from networks.Theta import Theta

    # Path to save/load pre-trained weights
    # Use a dedicated directory to avoid deletion by DataGenerator
    pretrained_dir = "../pretrained_models"
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)
        
    cache_filename = f"pretrained_{self.params.problem}_{self.params.case_name}_{self.params.init}.npy"
    cache_path = os.path.join(pretrained_dir, cache_filename)
            
    # If cache exists and we are not forcing regeneration
    if os.path.exists(cache_path) and not self.params.utils["gen_flag"]:
      print(f"Loading cached pre-trained weights from {cache_path}...")
      try:
        loaded_values = np.load(cache_path, allow_pickle=True)
        # Convert numpy arrays back to tensors
        theta_values = [
            tf.convert_to_tensor(v, dtype=tf.float32) for v in loaded_values
        ]
        self.model.nn_params = Theta(theta_values)
        print("Pre-training skipped (loaded from cache).")
        return
      except Exception as e:
        print(f"Failed to load cache: {e}. Re-running pre-training.")

    print(f"Pre-training phase with method {self.params.init}...")
    alg = self.__algorithm(self.params.init, self.params.param_init)
    alg.train()

    # Explicitly save history
    final_loss_mse = self.model.history[0]["Total"][-1] if self.model.history[
        0].get("Total") else "N/A"
    final_loss_llk = self.model.history[1]["Total"][-1] if self.model.history[
        1].get("Total") else "N/A"
    print(
        f"Pre-training finished. Final Loss (MSE): {final_loss_mse}, (LogLikelihood): {final_loss_llk}"
    )

    self.model.nn_params = self.model.thetas.pop()

    # Save weights to cache
    try:
      # Convert tensors to numpy for saving
      # Handle both Tensor (has .numpy()) and numpy array/EagerTensor cases
      numpy_values = []
      for v in self.model.nn_params.values:
        if hasattr(v, 'numpy'):
          numpy_values.append(v.numpy())
        else:
          numpy_values.append(np.array(v))

      np.save(cache_path, np.array(numpy_values, dtype=object))
      print(f"Saved pre-trained weights to {cache_path}")
    except Exception as e:
      print(f"Failed to save cache: {e}")

  def train(self):
    # Update active losses: use method-specific override or restore default
    if "losses" in self.params.param_method:
      print(f"Overriding losses for training: {self.params.param_method['losses']}")
      self.model.update_active_losses(self.params.param_method["losses"])
    else:
      # Restore default losses from global config (in case pre-train changed them)
      self.model.update_active_losses(self.params.losses)

    print(f"Training phase with method {self.params.method}...")
    alg = self.__algorithm(self.params.method, self.params.param_method)
    
    # Resume from checkpoint if available
    if self.plotter:
        import os
        import tensorflow as tf
        import numpy as np
        from networks.Theta import Theta
        
        base_path = os.path.dirname(self.plotter.path_plot)
        ckpt_path = os.path.join(base_path, "checkpoints", "checkpoint_latest.npy")
        
        if os.path.exists(ckpt_path):
            print(f"Resuming training from checkpoint: {ckpt_path}")
            try:
                loaded_values = np.load(ckpt_path, allow_pickle=True)
                theta_values = [tf.convert_to_tensor(v, dtype=tf.float32) for v in loaded_values]
                alg.model.nn_params = Theta(theta_values)
                print("Checkpoint loaded successfully.")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")

    alg.train()