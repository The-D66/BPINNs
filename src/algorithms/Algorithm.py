from utility import compute_gui_len
from setup import BatcherDataset
from abc import ABC, abstractmethod
from tqdm import tqdm
import time, datetime
import os
import numpy as np
import tensorflow as tf


class Algorithm(ABC):
  """
    Template for training
    """
  def __init__(self, bayes_nn, param_method, debug_flag):

    self.t0 = time.time()
    self.model = bayes_nn
    self.params = param_method
    self.epochs = self.params["epochs"]
    self.debug_flag = debug_flag
    self.curr_ep = 0

  def compute_time(self):
    training_time = time.time() - self.t0
    formatted_time = str(datetime.timedelta(seconds=int(training_time)))
    print(f'	Finished in {formatted_time}')

  @property
  def data_train(self):
    return self.__data

  @data_train.setter
  def data_train(self, dataset):
    self.model.u_coeff = dataset.norm_coeff["sol_mean"], dataset.norm_coeff[
        "sol_std"]
    self.model.f_coeff = dataset.norm_coeff["par_mean"], dataset.norm_coeff[
        "par_std"]
    self.__data = dataset  # Corrected: Assign to internal variable

  def __train_step(self):
    next(self.data_batch)
    match type(self).__name__:
      case "ADAM":
        new_theta = self.sample_theta(self.model.nn_params)
      case "HMC":
        new_theta = self.sample_theta(self.model.nn_params)
      case "SVGD":
        new_theta = self.sample_theta()
      case "VI":
        new_theta = self.sample_theta()
      case _:
        raise Exception("Method not Implemented!")
    self.__update_history(new_theta, type(self).__name__ == "SVGD")
    return new_theta

  def __train_loop(self, epochs):
    epochs_loop = range(epochs)
    if not self.debug_flag:
      epochs_loop = tqdm(epochs_loop)
      epochs_loop.ncols = compute_gui_len()
      epochs_loop.set_description_str("Training Progress")
    return epochs_loop

  def __generate_complex_plots(self):
    """ Generates time series and error distribution plots during training """
    try:
      # --- Time Series ---
      # Physical scales
      # self.params is param_method (e.g. HMC params). Global params are in self.model? No.
      # But we injected self.data_config in Trainer.
      if not hasattr(self, 'data_config'):
        return

      physics = self.data_config.physics
      L = physics["length"]
      T = physics["time"]

      target_locs_km = [0, 2, 4, 6, 8, 10]
      loc_data_list = []

      num_t = 100
      t_h_max = T / 3600.0
      t_h = np.linspace(0, t_h_max, num_t)
      t_norm = t_h / t_h_max

      for x_km in target_locs_km:
        x_norm_val = (x_km * 1000.0) / L
        x_col = np.full(num_t, x_norm_val)
        query_points = np.stack([x_col, t_norm], axis=1)

        preds = self.model.mean_and_std(query_points)

        # Exact values
        # Need to convert query_points to tensor for data_config.values
        query_points_tf = tf.constant(query_points.T, dtype=tf.float32)
        exact_res = self.data_config.values["u"](query_points_tf)
        h_ex = exact_res[0].flatten()
        u_ex = exact_res[1].flatten()

        # Training data near this location
        # We need to search in self.data_train.data_bnd/sol["dom"] (normalized)
        # Similar logic to main_plot_only.py
        train_pts_t = []
        train_pts_h = []
        train_pts_Q = []
        x_tol = 0.01

        # Normalization coefficients
        h_mean = self.data_train.norm_coeff["sol_mean"][0]
        u_mean = self.data_train.norm_coeff["sol_mean"][1]
        h_std = self.data_train.norm_coeff["sol_std"][0]
        u_std = self.data_train.norm_coeff["sol_std"][1]

        for name, data_dict in [
            ("sol", self.data_train.data_sol),
            ("bnd", self.data_train.data_bnd)
        ]:
          if data_dict["dom"].shape[0] > 0:
            mask = np.abs(data_dict["dom"][:, 0] - x_norm_val) < x_tol
            if np.any(mask):
              t_found = data_dict["dom"][mask, 1] * t_h_max
              # data_dict["sol"] is normalized [h, u]
              h_norm_val = data_dict["sol"][mask, 0]
              u_norm_val = data_dict["sol"][mask, 1]

              # Denormalize
              h_found = h_norm_val * h_std + h_mean
              u_found = u_norm_val * u_std + u_mean

              train_pts_t.extend(t_found)
              train_pts_h.extend(h_found)
              train_pts_Q.extend(h_found * u_found)

        loc_data = {
            "x_km":
                x_km,
            "t_h":
                t_h,
            "h_ex":
                h_ex,
            "h_mean":
                preds["sol_NN"][:, 0],
            "h_std":
                preds["sol_std"][:, 0],
            "Q_ex":
                h_ex * u_ex,
            "Q_mean":
                preds["sol_NN"][:, 0] * preds["sol_NN"][:, 1],
            "Q_std":
                np.sqrt(
                    (preds["sol_NN"][:, 1] * preds["sol_std"][:, 0])**2 +
                    (preds["sol_NN"][:, 0] * preds["sol_std"][:, 1])**2
                ),
            "train_t":
                np.array(train_pts_t),
            "train_h":
                np.array(train_pts_h),
            "train_Q":
                np.array(train_pts_Q)
        }
        loc_data_list.append(loc_data)

      self.plotter.plot_time_series(loc_data_list)

      # --- Error Distribution ---
      all_train_coords_list = []
      if self.data_train.data_sol["dom"].shape[0] > 0:
        all_train_coords_list.append(self.data_train.data_sol["dom"])
      if self.data_train.data_bnd["dom"].shape[0] > 0:
        all_train_coords_list.append(self.data_train.data_bnd["dom"])

      if len(all_train_coords_list) > 0:
        all_train_coords = np.concatenate(all_train_coords_list, axis=0)
        preds_train = self.model.mean_and_std(all_train_coords)

        all_train_coords_tf = tf.constant(all_train_coords.T, dtype=tf.float32)
        h_ex_train, u_ex_train = self.data_config.values["u"](
            all_train_coords_tf
        )

        train_data_for_plot = {
            "coords": all_train_coords,
            "h_ex": h_ex_train,
            "u_ex": u_ex_train,
            "h_nn": preds_train['sol_NN'][:, 0],
            "u_nn": preds_train['sol_NN'][:, 1]
        }
        self.plotter.plot_error_distribution(train_data_for_plot)

    except Exception as e:
      print(f"Complex plotting failed: {e}")

  def train(self):

    # Store thetas in this round of training
    thetas_train = list()
    # Normalizing dataset
    self.data_train.normalize_dataset()
    self.model.norm_coeff = self.data_train.norm_coeff
    self.data_batch = BatcherDataset(self.data_train, num_batch=1)

    # Sampling new thetas
    self.epochs_loop = self.__train_loop(self.epochs)
    for i in self.epochs_loop:
      if self.debug_flag:
        print(f'  START EPOCH {i+1}')
      self.curr_ep = i + 1
      step = self.__train_step()
      thetas_train.append(step)

      # Update tqdm description with current loss
      if not self.debug_flag and hasattr(
          self.epochs_loop, 'set_description_str'
      ):
        # history is (mse, loglikelihood)
        # Ensure all keys are present before trying to access
        loss_info = {}
        for key in [
            "Total", "data_u", "data_b", "pde", "prior"
        ]:  # Explicitly list the keys to display
          loss_info[key] = self.model.history[0][key][-1] if self.model.history[
              0].get(key) else 0.0

        desc_str = f"Training Progress (Total: {loss_info['Total']:.2e}"
        if loss_info.get("data_u") is not None:
          desc_str += f" | data_u: {loss_info['data_u']:.2e}"
        if loss_info.get("data_b") is not None:
          desc_str += f" | data_b: {loss_info['data_b']:.2e}"
        if loss_info.get("pde") is not None:
          desc_str += f" | pde: {loss_info['pde']:.2e}"
        if loss_info.get("prior") is not None:
          desc_str += f" | prior: {loss_info['prior']:.2e}"
        desc_str += ")"
        self.epochs_loop.set_description_str(desc_str)

      # Real-time plotting and saving (conditional frequency)
      plot_freq = 500 if type(self).__name__ == "ADAM" else 100
      if (i + 1) % plot_freq == 0 and hasattr(
          self, 'plotter'
      ) and self.plotter is not None:
        # Save Checkpoint
        try:
          # Checkpoint path: ../outs/.../checkpoints/
          base_path = os.path.dirname(self.plotter.path_plot)
          ckpt_dir = os.path.join(base_path, "checkpoints")
          if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
          ckpt_path = os.path.join(ckpt_dir, "checkpoint_latest.npy")
          # Save nn_params values (list of numpy arrays)
          numpy_values = []
          for v in self.model.nn_params.values:
            if hasattr(v, 'numpy'):
              numpy_values.append(v.numpy())
            else:
              numpy_values.append(np.array(v))
          np.save(ckpt_path, np.array(numpy_values, dtype=object))
        except Exception as e:
          pass  # Ignore save errors during training loop to avoid stopping

        # Plotting
        try:
          # Backup original thetas
          backup_thetas = self.model.thetas

          # Use collected samples so far for visualization
          if type(self.model).__name__ == "ADAM":
            # For ADAM, we want to see the CURRENT model state, not the trajectory mean.
            self.model.thetas = [self.model.nn_params]
          elif thetas_train:
            # For HMC/VI, we want to see the posterior distribution accumulated so far.
            self.model.thetas = thetas_train
          else:
            self.model.thetas = [self.model.nn_params] # Fallback if no samples yet

          # Set subfolder for intermediate plots
          self.plotter.set_subfolder(f"epoch_{i+1}")

          # Predict on test domain (normalized inputs)
          # self.data_train.data_test["dom"] is normalized in place
          functions = self.model.mean_and_std(self.data_train.data_test["dom"])

          # self.test_data_phys contains physical exact values (backed up in Trainer)
          self.plotter.plot_confidence(self.test_data_phys, functions)
          self.plotter.plot_full_domain_error(self.test_data_phys, functions)

          # Plot History (reconstruct history tuple from model)
          # self.model.history is (pst, llk)
          self.plotter.plot_losses(self.model.history)

          # Generate complex plots (Time Series & Error Dist)
          self.__generate_complex_plots()

          # Reset subfolder
          self.plotter.set_subfolder(None)

          # Restore thetas
          self.model.thetas = backup_thetas

        except Exception as e:
          print(f"\n[Warning] Plotting failed at epoch {i+1}: {e}")

    # Denormalizing dataset
    self.data_train.denormalize_dataset()
    # Select which thetas must be saved
    thetas_train = self.select_thetas(thetas_train)
    # Save thetas in the bnn
    self.model.thetas += thetas_train
    # Report training information
    self.train_log()

  def train_log(self):
    """ Report log of the training"""
    print('End training:')
    self.compute_time()

  def __update_history(self, new_theta, svgd_flag=False):
    # Saving new theta
    self.model.nn_params = new_theta if not svgd_flag else new_theta[-1]
    # Computing History
    pst, llk = self.model.metric_total(self.data_batch)
    self.model.loss_step((pst, llk))

  @abstractmethod
  def sample_theta(self):
    """ 
        Method for sampling a single new theta
        Must be overritten in child classes
        """
    return None

  @abstractmethod
  def select_thetas(self, thetas_train):
    """ 
        Compute burn-in and skip samples
        Must be overritten in child classes
        """
    return list()