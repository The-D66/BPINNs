import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

class Plotter():
    """ 
    Class for plotting utilities:
    Methods:
        - plot_losses: plots MSE and log-likelihood History
        - plot_nn_samples: plots all samples of solution and parametric field
        - plot_confidence: plots mean and std of solution and parametric field
        - show_plot: enables plot visualization
    """
    
    def __init__(self, path_folder):
        
        self.path_plot = os.path.join(path_folder,"plot")
        self.path_log  = os.path.join(path_folder,"log")
        with open(os.path.join(self.path_log,"parameters.txt"), "r") as file_params:
            lines = file_params.readlines()
            problem = lines[2][10:].strip()
            
        self.only_sol = problem == "Regression" or problem == "Oscillator"
        
        # Physical scaling for SaintVenant1D
        if "SaintVenant" in problem:
            self.scale_x = 10.0 # 10 km
            self.scale_t = 4.0  # 14400s = 4h
            self.label_x = "x (km)"
            self.label_t = "t (h)"
        else:
            self.scale_x = 1.0
            self.scale_t = 1.0
            self.label_x = "x"
            self.label_t = "t"
            
        self.subfolder = None

    def set_subfolder(self, name):
        """ Set a subfolder for saving plots (e.g. for checkpoints) """
        self.subfolder = name

    def __order_inputs(self, inputs):
        """ Sorting the input points by label """
        idx = np.argsort(inputs)
        inputs = inputs[idx]
        return inputs, idx

    def __save_plot(self, path, title):
        """ Auxiliary function used in all plot functions for saving """
        save_path = path
        if self.subfolder:
            save_path = os.path.join(save_path, self.subfolder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        
        file_path = os.path.join(save_path, title)
        plt.savefig(file_path, bbox_inches = 'tight')
        plt.close()

    def __plot_confidence_1D(self, x, func, title, label = ("",""), fit = None):
        """ Plots mean and standard deviation of func (1D case); used in plot_confidence """
        x, idx = self.__order_inputs(x)
        func = [f[idx] for f in func]

        plt.figure()
        plt.plot(x, func[0], 'r-',  label='true')
        plt.plot(x, func[1], 'b--', label='mean')
        plt.plot(x, func[1] - func[2], 'g--', label='mean-std')
        plt.plot(x, func[1] + func[2], 'g--', label='mean+std')
        if fit is not None:
            plt.plot(fit[0], fit[1], 'r*')

        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.legend(prop={'size': 9})
        plt.title(title)

    def __plot_confidence_2D(self, x, func, title, label = ("",""), fit = None):
        """ Plots mean and standard deviation of func (2D case); used in plot_confidence """
        # Check if vector output (N, dim_out)
        func_ex = func[0]
        if len(func_ex.shape) > 1 and func_ex.shape[1] > 1:
            # Vector output: loop over components
            dim_out = func_ex.shape[1]
            component_names = ["h", "u"] if dim_out == 2 else [f"comp_{i}" for i in range(dim_out)]
            
            for i in range(dim_out):
                # Slice component i
                # func is tuple (exact, mean, std)
                # exact: (N, dim), mean: (N, dim), std: (N, dim)
                # We need to handle if std is scalar or vector. Usually vector.
                
                f_ex = func[0][:, i]
                f_mn = func[1][:, i]
                f_sd = func[2][:, i]
                
                comp_func = (f_ex, f_mn, f_sd)
                comp_title = f"{title}_{component_names[i]}"
                
                # Fit data: fit is usually (dom, sol). sol might be (N, dim).
                comp_fit = None
                if fit is not None:
                    # fit[1] is sol.
                    if len(fit[1].shape) > 1 and fit[1].shape[1] > i:
                         comp_fit = (fit[0], fit[1][:, i])
                
                self.__plot_confidence_2D_scalar(x, comp_func, comp_title, label, comp_fit)
                self.__save_plot(self.path_plot, f"{comp_title}.png")
        else:
            self.__plot_confidence_2D_scalar(x, func, title, label, fit)
            self.__save_plot(self.path_plot, f"{title}.png")

    def __plot_confidence_2D_scalar(self, x, func, title, label = ("",""), fit = None):
        """ Plots mean and standard deviation of func (2D case) - Scalar Version """

        N = len(func[0]) # Corrected: func[0] is the exact solution for scalar component
        xx = np.unique(x[:,0])
        xx, _ = self.__order_inputs(xx)
        
        # Scale coordinates
        xx = xx * self.scale_x

        yy = np.unique(x[:,1])
        yy, _ = self.__order_inputs(yy)
        
        # Scale coordinates
        yy = yy * self.scale_t

        # Create meshgrid for plotting
        X, Y = np.meshgrid(xx, yy)


        func_ex = func[0]
        func_nn = func[1]

        # Reshape assuming grid structure
        num_x_unique = len(xx)
        num_y_unique = len(yy)

        if num_x_unique * num_y_unique == N:
            # Reshape based on number of unique x and y points
            # Assuming func_ex elements are ordered consistently with meshgrid's flatten (x varies fastest, t slowest)
            # Reshaping to (num_y_unique, num_x_unique) means y (t) is the row index, x is the column index
            Z_ex = np.reshape(func_ex, [num_y_unique, num_x_unique])
            Z_nn = np.reshape(func_nn, [num_y_unique, num_x_unique])
            
            print(f"--- Plotting {title} ---")
            print(f"  Exact: Min={np.min(Z_ex):.4f}, Max={np.max(Z_ex):.4f}, Mean={np.mean(Z_ex):.4f}")
            print(f"  NN   : Min={np.min(Z_nn):.4f}, Max={np.max(Z_nn):.4f}, Mean={np.mean(Z_nn):.4f}")
            
        else:
            print(f"Warning: Data size {N} is not compatible with grid {num_x_unique}x{num_y_unique}, skipping pcolor plot for {title}")
            return

        # Compute global min/max for shared colorbar
        g_min = min(np.min(Z_ex), np.min(Z_nn))
        g_max = max(np.max(Z_ex), np.max(Z_nn))
        
        # Manual override for better visualization of SaintVenant
        if "h" in title: # Water depth
             # Auto-scale but ensure it covers reasonable range
             # g_min = max(0, g_min) # h shouldn't be negative
             pass
        if "u" in title and "h" not in title: # Velocity
             pass

        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12, 5))
        # Adjust layout to make room for colorbar on the right
        fig.subplots_adjust(right=0.85, wspace=0.3)
        
        # First subplot - Exact solution
        p1 = ax1.pcolor(X, Y, Z_ex, cmap='viridis', vmin=g_min, vmax=g_max)
        ax1.set_xlabel(self.label_x)
        ax1.set_ylabel(self.label_t)
        ax1.set_title(f"Exact Solution")

        # Second subplot - NN solution
        p2 = ax2.pcolor(X, Y, Z_nn, cmap='viridis', vmin=g_min, vmax=g_max)
        ax2.set_xlabel(self.label_x)
        ax2.set_ylabel(self.label_t)
        ax2.set_title(f"Predicted Mean")

        # Shared Colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
        cb = fig.colorbar(p2, cax=cbar_ax)
        cb.set_label(f"{title}") # Set colorbar label

    def __plot_nn_samples_1D(self, x, func, label = ("",""), fit = None):
        """ Plots all the samples of func; used in plot_nn_samples """
        x, idx = self.__order_inputs(x)

        plt.figure()
        blurring = min(1.0, 2/len(func[1]))
        for func_sample in func[1]:
            plt.plot(x, func_sample[idx,0], 'b-', markersize=0.01, alpha=blurring)

        func_ex = func[0][idx]
        plt.plot(x, func_ex, 'r-', label='true')
        if fit is not None:
            plt.plot(fit[0], fit[1], 'r*')

        plt.xlabel(label[0])
        plt.ylabel(label[1])
        plt.legend(prop={'size': 9})
        plt.title('Samples from ' + label[1] + ' reconstructed distribution')

    def __plot_train(self, losses, name, title):
        """ Plots all the loss history; used in plot_losses """
        plt.figure()
        x = list(range(1,len(losses['Total'])+1))
        if name[:-4] == "LogLoss":
            plt.semilogx(x, losses['Total'], 'k--', lw=2.0, alpha=1.0, label = 'Total')
        for key, value in losses.items():
            if key == "Total": continue
            plt.semilogx(x, value, lw=1.0, alpha=0.7, label = key)
        plt.title(f"History of {title}")
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.legend(prop={'size': 9})
        self.__save_plot(self.path_plot, title)

    def plot_confidence(self, data, functions):
        """ Plots mean and standard deviation of solution and parametric field samples """
        if data["sol_ex"][0].shape[1] == 1:
            x = (data["sol_ex"][0][:,0], data["par_ex"][0][:,0])
            u = (data["sol_ex"][1], functions['sol_NN'], functions['sol_std'])
            f = (data["par_ex"][1], functions['par_NN'], functions['par_std'])

            self.__plot_confidence_1D(x[0], u, 'Confidence interval for u(x)', label = ('x','u'), fit = data["sol_ns"])
            self.__save_plot(self.path_plot, 'u_confidence.png')
            if self.only_sol: return
            
            # Check if par data exists
            if f[1].size > 0:
                self.__plot_confidence_1D(x[1], f, 'Confidence interval for f(x)', label = ('x','f'), fit = data["par_ns"])
                self.__save_plot(self.path_plot, 'f_confidence.png')

        elif data["sol_ex"][0].shape[1] == 2:
            x = (data["sol_ex"][0], data["par_ex"][0])
            u = (data["sol_ex"][1], functions['sol_NN'], functions['sol_std'])
            f = (data["par_ex"][1], functions['par_NN'], functions['par_std'])

            # Renamed save inside the method
            self.__plot_confidence_2D(x[0], u, 'u', label = ('x','u'), fit = data["sol_ns"])
            # self.__save_plot(self.path_plot, 'u_confidence.png') # Handled inside
            
            if self.only_sol: return
            
            # Check if par data exists
            if f[1].size > 0:
                self.__plot_confidence_2D(x[1], f, 'f', label = ('x','f'), fit = data["par_ns"])
                # self.__save_plot(self.path_plot, 'f_confidence.png')

    def plot_nn_samples(self, data, functions):
        """ Plots all the samples of solution and parametric field """
        if data["sol_ex"][0].shape[1] == 1 :
            x = (data["sol_ex"][0][:,0], data["par_ex"][0][:,0])
            u = (data["sol_ex"][1], functions['sol_samples'])
            f = (data["par_ex"][1], functions['par_samples'])

            self.__plot_nn_samples_1D(x[0], u, label = ('x','u'), fit = data["sol_ns"])
            self.__save_plot(self.path_plot, 'u_nn_samples.png')
            if self.only_sol: return
            self.__plot_nn_samples_1D(x[1], f, label = ('x','f'), fit = data["par_ns"])
            self.__save_plot(self.path_plot, 'f_nn_samples.png')
        else: 
            pass 

    def plot_losses(self, losses):
        """ Generates the plots of MSE and log-likelihood """
        self.__plot_train(losses[0], "Loss.png"   , "Mean Squared Error")
        # Duplicate for specific user request
        self.__save_plot(self.path_plot, "Mean Squared Error.png") 
        self.__plot_train(losses[1], "LogLoss.png", "Loss (Log-Likelihood)")
        # Rename to match request exactly
        self.__save_plot(self.path_plot, "Loss (Log-Likelihood).png")

    def plot_all(self, history, data, functions, train_data_for_plot=None, loc_data_list=None, pde_data=None, bnd_data=None, sol_data=None):
        """ 
        Unified method to generate all requested plots with error handling 
        """
        print("Generating all plots...")
        
        # 1. Loss Plots
        try:
            if history: self.plot_losses(history)
        except Exception as e: print(f"Failed to plot losses: {e}")

        # 2. Confidence Plots (u_h.png, u_u.png)
        try:
            if data and functions: self.plot_confidence(data, functions)
        except Exception as e: print(f"Failed to plot confidence: {e}")

        # 3. Full Domain Error (full_domain_error.png)
        try:
            if data and functions: self.plot_full_domain_error(data, functions)
        except Exception as e: print(f"Failed to plot full domain error: {e}")

        # 4. Training Data Distribution (Optional/Debug)
        try:
            # Re-use provided data if available
            if pde_data or bnd_data or sol_data:
                self.plot_training_data(data, pde_data, bnd_data, sol_data)
        except Exception as e: print(f"Failed to plot training data dist: {e}")

        # 5. Error Distribution on Training Points (error_distribution_training_points.png)
        try:
            if train_data_for_plot: self.plot_error_distribution(train_data_for_plot)
        except Exception as e: print(f"Failed to plot error distribution: {e}")

        # 6. Time Series (time_series_h.png, time_series_Q.png)
        try:
            if loc_data_list: self.plot_time_series(loc_data_list)
        except Exception as e: print(f"Failed to plot time series: {e}")

    def plot_training_data(self, data, pde_data, bnd_data, sol_data):
        """ Plots the distribution of training points (sol, bnd, pde) """
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        ax.set_xlabel(self.label_x)
        ax.set_ylabel(self.label_t)
        ax.set_title("Training Data Distribution")
        ax.set_xlim([0, self.scale_x])
        ax.set_ylim([0, self.scale_t])

        # Plot PDE points
        if pde_data is not None and len(pde_data["dom"]) > 0:
            ax.scatter(pde_data["dom"][:,0] * self.scale_x, pde_data["dom"][:,1] * self.scale_t, 
                        s=5, alpha=0.05, color='gray', label='PDE Points')

        # Plot Boundary points
        if bnd_data is not None and len(bnd_data["dom"]) > 0:
            ax.scatter(bnd_data["dom"][:,0] * self.scale_x, bnd_data["dom"][:,1] * self.scale_t, 
                        s=20, alpha=0.3, color='blue', label='Boundary Points')
        
        # Plot Solution points
        if sol_data is not None and len(sol_data["dom"]) > 0:
            ax.scatter(sol_data["dom"][:,0] * self.scale_x, sol_data["dom"][:,1] * self.scale_t, 
                        s=40, alpha=0.5, color='green', marker='x', label='Solution Points (h)')
        
        ax.legend(loc='upper right', markerscale=2)
        
        self.__save_plot(self.path_plot, "training_data_distribution.png")

    def plot_qq(self, data, functions):
        """ Plots QQ plots for h and u to compare Exact vs NN distribution """
        
        # Exact solution
        # data["sol_ex"] is (dom, val). val is (N, 2)
        h_ex = data["sol_ex"][1][:,0].flatten()
        u_ex = data["sol_ex"][1][:,1].flatten()
        
        # NN Prediction
        # functions['sol_NN'] is (N, 2)
        h_nn = functions['sol_NN'][:,0].flatten()
        u_nn = functions['sol_NN'][:,1].flatten()
        
        # Sort for QQ plot
        h_ex_sorted = np.sort(h_ex)
        h_nn_sorted = np.sort(h_nn)
        u_ex_sorted = np.sort(u_ex)
        u_nn_sorted = np.sort(u_nn)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # h QQ Plot
        ax1.scatter(h_ex_sorted, h_nn_sorted, s=5, alpha=0.2, color='blue')
        # Reference line
        min_val = min(np.min(h_ex), np.min(h_nn))
        max_val = max(np.max(h_ex), np.max(h_nn))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
        ax1.set_xlabel("Exact h")
        ax1.set_ylabel("Predicted h")
        ax1.set_title("QQ Plot: Water Depth (h)")
        ax1.grid(True, alpha=0.3)
        
        # u QQ Plot
        ax2.scatter(u_ex_sorted, u_nn_sorted, s=5, alpha=0.2, color='red')
        # Reference line
        min_val = min(np.min(u_ex), np.min(u_nn))
        max_val = max(np.max(u_ex), np.max(u_nn))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
        ax2.set_xlabel("Exact u")
        ax2.set_ylabel("Predicted u")
        ax2.set_title("QQ Plot: Velocity (u)")
        ax2.grid(True, alpha=0.3)
        
        self.__save_plot(self.path_plot, "qq_plot.png")

    def plot_full_domain_error(self, data, functions):
        """ Plots the error distribution on the full test domain using pcolor and histograms """
        
        # Extract coordinates and values
        x = data["sol_ex"][0]
        xx = np.unique(x[:,0])
        yy = np.unique(x[:,1])
        
        # Sort for reshaping
        xx, _ = self.__order_inputs(xx)
        yy, _ = self.__order_inputs(yy)
        
        # Scale coords for plotting
        X_mesh, Y_mesh = np.meshgrid(xx * self.scale_x, yy * self.scale_t)
        
        # Exact values
        h_ex = data["sol_ex"][1][:,0]
        u_ex = data["sol_ex"][1][:,1]
        
        # NN values
        h_nn = functions['sol_NN'][:,0]
        u_nn = functions['sol_NN'][:,1]
        
        # Errors
        err_h = h_nn - h_ex
        err_u = u_nn - u_ex
        
        # Calculate RMSE
        rmse_h = np.sqrt(np.mean(err_h**2))
        rmse_u = np.sqrt(np.mean(err_u**2))
        
        # Reshape
        N = len(h_ex)
        if len(xx) * len(yy) == N:
            Z_err_h = np.reshape(err_h, [len(yy), len(xx)])
            Z_err_u = np.reshape(err_u, [len(yy), len(xx)])
        else:
            print(f"Warning: Full domain error plot skipped due to shape mismatch ({N} vs {len(xx)}x{len(yy)})")
            return

        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.tight_layout(pad=4.0)
        
        # Helper for symmetric limits
        def get_limit(arr):
            m = np.max(np.abs(arr))
            return m if m > 1e-8 else 0.1
            
        lim_h = get_limit(Z_err_h)
        lim_u = get_limit(Z_err_u)
        
        # --- Row 1: Water Depth h ---
        
        # Map
        ax1 = axes[0, 0]
        p1 = ax1.pcolor(X_mesh, Y_mesh, Z_err_h, cmap='RdBu_r', vmin=-lim_h, vmax=lim_h)
        cb1 = fig.colorbar(p1, ax=ax1)
        cb1.set_label("Error (m)")
        ax1.set_xlabel(self.label_x)
        ax1.set_ylabel(self.label_t)
        ax1.set_title("Water Depth Error Distribution")
        
        # Histogram
        ax2 = axes[0, 1]
        ax2.hist(err_h, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel("Error Value (m)")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Water Depth Error Histogram\nRMSE = {rmse_h:.4e} m")
        ax2.grid(True, alpha=0.3)
        
        # --- Row 2: Velocity u ---
        
        # Map
        ax3 = axes[1, 0]
        p2 = ax3.pcolor(X_mesh, Y_mesh, Z_err_u, cmap='RdBu_r', vmin=-lim_u, vmax=lim_u)
        cb2 = fig.colorbar(p2, ax=ax3)
        cb2.set_label("Error (m/s)")
        ax3.set_xlabel(self.label_x)
        ax3.set_ylabel(self.label_t)
        ax3.set_title("Velocity Error Distribution")
        
        # Histogram
        ax4 = axes[1, 1]
        ax4.hist(err_u, bins=50, color='salmon', edgecolor='black', alpha=0.7)
        ax4.set_xlabel("Error Value (m/s)")
        ax4.set_ylabel("Frequency")
        ax4.set_title(f"Velocity Error Histogram\nRMSE = {rmse_u:.4e} m/s")
        ax4.grid(True, alpha=0.3)
        
        self.__save_plot(self.path_plot, "full_domain_error.png")

    def plot_error_distribution(self, train_data_for_plot):
        """ Plots the spatial distribution of errors for h and u on training points """
        
        # Unpack data
        x_phys = train_data_for_plot["coords"][:,0] * self.scale_x
        t_phys = train_data_for_plot["coords"][:,1] * self.scale_t
        
        h_ex = train_data_for_plot["h_ex"].flatten()
        u_ex = train_data_for_plot["u_ex"].flatten()
        h_nn = train_data_for_plot["h_nn"].flatten()
        u_nn = train_data_for_plot["u_nn"].flatten()
        
        # Errors (Predicted - Exact)
        err_h = h_nn - h_ex
        err_u = u_nn - u_ex
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.tight_layout(pad=3.0)
        
        # Helper for symmetric colorbar
        def get_limit(arr):
            m = np.max(np.abs(arr))
            return m if m > 1e-8 else 0.1 # Avoid 0 limit
            
        limit_h = get_limit(err_h)
        limit_u = get_limit(err_u)
        
        # Plot h Error
        p1 = ax1.scatter(x_phys, t_phys, c=err_h, cmap='RdBu_r', s=20, alpha=0.8, vmin=-limit_h, vmax=limit_h)
        cb1 = fig.colorbar(p1, ax=ax1)
        cb1.set_label("Error (h_pred - h_ex)")
        ax1.set_xlabel(self.label_x)
        ax1.set_ylabel(self.label_t)
        ax1.set_title("Water Depth Error Distribution (Training Points)")
        
        # Plot u Error
        p2 = ax2.scatter(x_phys, t_phys, c=err_u, cmap='RdBu_r', s=20, alpha=0.8, vmin=-limit_u, vmax=limit_u)
        cb2 = fig.colorbar(p2, ax=ax2)
        cb2.set_label("Error (u_pred - u_ex)")
        ax2.set_xlabel(self.label_x)
        ax2.set_ylabel(self.label_t)
        ax2.set_title("Velocity Error Distribution (Training Points)")
        
        self.__save_plot(self.path_plot, "error_distribution_training_points.png")

    def plot_time_series(self, loc_data_list):
        """ 
        Plots time series of h and Q at specific locations 
        loc_data_list: list of dicts with keys: x_km, t_h, h_ex, h_mean, h_std, Q_ex, Q_mean, Q_std
        """
        
        num_locs = len(loc_data_list)
        cols = 2
        rows = (num_locs + 1) // cols
        
        # --- Plot Water Depth h ---
        fig_h, axes_h = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=True, sharey=True)
        axes_h = axes_h.flatten()
        
        for i, data in enumerate(loc_data_list):
            ax = axes_h[i]
            t = data["t_h"]
            
            # Exact
            ax.plot(t, data["h_ex"], 'k-', linewidth=1.5, label='Exact')
            
            # Training Data (Noisy)
            if len(data["train_t"]) > 0:
                ax.scatter(data["train_t"], data["train_h"], color='red', marker='x', s=20, label='Noisy Data', zorder=10)
            
            # NN Mean
            ax.plot(t, data["h_mean"], 'b--', linewidth=1.5, label='NN Mean')
            
            # Uncertainty (2 std)
            lower = data["h_mean"] - 2 * data["h_std"]
            upper = data["h_mean"] + 2 * data["h_std"]
            ax.fill_between(t, lower, upper, color='blue', alpha=0.2, label='95% CI')
            
            ax.set_title(f"x = {data['x_km']:.1f} km")
            ax.grid(True, alpha=0.3)
            
            # Only add legend to the first plot
            if i == 0: ax.legend(loc='best', fontsize='small')

        # Set global labels
        fig_h.supxlabel("Time (h)")
        fig_h.supylabel("Water Depth h (m)")

        # Hide empty subplots
        for i in range(num_locs, len(axes_h)):
            axes_h[i].axis('off')
            
        fig_h.tight_layout()
        self.__save_plot(self.path_plot, "time_series_h.png")
        
        # --- Plot Discharge Q ---
        fig_q, axes_q = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=True, sharey=True)
        axes_q = axes_q.flatten()
        
        for i, data in enumerate(loc_data_list):
            ax = axes_q[i]
            t = data["t_h"]
            
            # Exact
            ax.plot(t, data["Q_ex"], 'k-', linewidth=1.5, label='Exact')
            
            # Training Data (Noisy)
            if len(data["train_t"]) > 0:
                ax.scatter(data["train_t"], data["train_Q"], color='red', marker='x', s=20, label='Noisy Data', zorder=10)
            
            # NN Mean
            ax.plot(t, data["Q_mean"], 'b--', linewidth=1.5, label='NN Mean') # Changed to blue dash to match h style
            
            # Uncertainty (2 std)
            lower = data["Q_mean"] - 2 * data["Q_std"]
            upper = data["Q_mean"] + 2 * data["Q_std"]
            ax.fill_between(t, lower, upper, color='blue', alpha=0.2, label='95% CI') # Changed to blue shade
            
            ax.set_title(f"x = {data['x_km']:.1f} km")
            ax.grid(True, alpha=0.3)
            if i == 0: ax.legend(loc='best', fontsize='small')

        # Set global labels
        fig_q.supxlabel("Time (h)")
        fig_q.supylabel("Discharge Q (m³/s)")

        # Hide empty subplots
        for i in range(num_locs, len(axes_q)):
            axes_q[i].axis('off')
            
        fig_q.tight_layout()
        self.__save_plot(self.path_plot, "time_series_Q.png")

    def show_plot(self):
        """ Shows the plots """
        print(f"Plots saved to {self.path_plot}")