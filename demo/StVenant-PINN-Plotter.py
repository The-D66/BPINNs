#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil

# Disable GPU at start
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import timeit
from scipy import optimize
from postprocessing.Plotter import Plotter

# --- Finite Difference Solver Functions ---

def stvenant_system_midp_left(sol,sol_past,sol_left,x_left,dx,t_past,dt):
  """
  左侧中点格式 (Midpoint Left Scheme)
  """
  h = sol[0]; u=sol[1];
  h_past = sol_past[0]; u_past = sol_past[1];
  h_left = sol_left[0]; u_left = sol_left[1];
  
  dhdt = (h-h_past)/dt; dudt = (u-u_past)/dt;
  dhdx = (h-h_left)/dx; dudx = (u-u_left)/dx;
  
  # 在中点 (x_left + dx/2) 处计算系数
  mass = A2(x_left+dx/2,(h_left+h)/2)*dhdt    \
       + u*(A1(x_left+dx/2,(h_left+h)/2)      \
       + A2(x_left+dx/2,(h_left+h)/2)*dhdx)   \
       + Af(x_left+dx/2,(h_left+h)/2)*dudx
  momentum = dudt + (u_left+u)/2*dudx + g*dhdx            \
           + g*(Sf(x_left+dx/2,(h_left+h)/2,(u_left+u)/2) \
                -dzb(x_left+dx/2))
  return np.array([mass,momentum])

def solve_finite_difference(a_x, b_x, nx, final_t, nt, h_init_fn, u_init_fn, h_bc_fn, u_bc_fn):
    print("Running Finite Difference Solver for Exact Solution...")
    x = np.linspace(a_x, b_x, nx)
    t = np.linspace(0, final_t, nt)
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    
    H = np.zeros((len(x), len(t)))
    U = np.zeros((len(x), len(t)))

    # Initialization
    for i in range(len(x)):
        H[i,0] = h_init_fn(x[i])
        U[i,0] = u_init_fn(x[i])
    for j in range(1,len(t)):
        H[0,j] = h_bc_fn(t[j])
        U[0,j] = u_bc_fn(t[j])

    # Solver Loop
    for j in range(1,len(t)):
        for i in range(1,len(x)):
            sol_past = np.array([H[i,j-1],U[i,j-1]])
            sol_left = np.array([H[i-1,j],U[i-1,j]])
            sol_guess = sol_past
            
            # Optimization to solve implicit system
            # Pass current global functions implicitly via stvenant_system_midp_left which uses them
            output = optimize.root(stvenant_system_midp_left, sol_guess,
                                   args=(sol_past,sol_left,x[i-1],dx,t[j-1],dt),
                                   method='hybr', tol=1e-9)
            if not output.success:
                 # Fallback or warning
                 pass
            H[i,j] = output.x[0]
            U[i,j] = output.x[1]
            
    return x, t, H, U

# --- PINN Class and Functions ---

class PINN(tf.keras.Model):
    def __init__(self, neurons=16, layers=4, activation='tanh'):
        super(PINN, self).__init__()
        self.hidden_layers = []
        for _ in range(layers - 1):
            self.hidden_layers.append(tf.keras.layers.Dense(neurons, activation=activation))
        self.output_layer = tf.keras.layers.Dense(2)

    def call(self, x):
        z = x
        for layer in self.hidden_layers:
            z = layer(z)
        return self.output_layer(z)

def derivative(model, x, t, output_index=0):
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, t])
        xt = tf.concat([x, t], axis=1)
        uy = model(xt)
        out = uy[:, output_index:output_index+1]
        if output_index == 1:
            eps = 1e-1
            out = tf.where(out < 0.0, tf.nn.softplus(out) * eps, out)
    out_x = tape1.gradient(out, x)
    out_t = tape1.gradient(out, t)
    del tape1
    return out, out_x, out_t

def residual(model, x, t):
    u, u_x, u_t = derivative(model, x, t, 0)
    h, h_x, h_t = derivative(model, x, t, 1)
    continuity_residual = A2(x,h)*h_t + u*(A1(x,h) + A2(x,h)*h_x) + Af(x,h)*u_x
    momentum_residual = u_t + u*u_x + g*h_x + g*(Sf(x,h,u) - dzb(x))
    return continuity_residual, momentum_residual

def loss_fn(model, x, t, a_x, b_x):
    continuity_residual, momentum_residual = residual(model, x, t)
    mse_continuity = tf.reduce_mean(tf.square(continuity_residual))
    mse_momentum = tf.reduce_mean(tf.square(momentum_residual))

    t0 = tf.zeros_like(x)
    xt0 = tf.concat([x, t0], axis=1)
    u0_pred = model(xt0)[:, 0:1]
    h0_pred = model(xt0)[:, 1:2]
    u0_true = u_init(x)
    h0_true = h_init(x)
    mse_ic_u = tf.reduce_mean(tf.square(u0_pred - u0_true))
    mse_ic_h = tf.reduce_mean(tf.square(h0_pred - h0_true))

    xl = tf.ones_like(t) * a_x
    xlt = tf.concat([xl, t], axis=1)
    u_l_pred = model(xlt)[:, 0:1]
    h_l_pred = model(xlt)[:, 1:2]
    u_l_true = u_bcleft(t)
    h_l_true = h_bcleft(t)
    mse_bc_u = tf.reduce_mean(tf.square(u_l_pred - u_l_true))
    mse_bc_h = tf.reduce_mean(tf.square(h_l_pred - h_l_true))

    total_loss = mse_continuity + mse_momentum + mse_ic_u + mse_ic_h + mse_bc_u + mse_bc_h
    return total_loss

def find_mesh_subintervals(xn_pts, tn_pts, a_x, b_x, final_t):
    x_edges = tf.linspace(a_x, b_x, xn_pts + 1)
    t_edges = tf.linspace(0.0, final_t, tn_pts + 1)
    x_offsets = tf.random.uniform((xn_pts,), 0, 1, dtype=tf.float32)
    t_offsets = tf.random.uniform((tn_pts,), 0, 1, dtype=tf.float32)
    x_points = x_edges[:-1] + x_offsets * (x_edges[1:] - x_edges[:-1])
    t_points = t_edges[:-1] + t_offsets * (t_edges[1:] - t_edges[:-1])
    x_sorted = tf.sort(x_points, axis=0)
    t_sorted = tf.sort(t_points, axis=0)
    X, T = tf.meshgrid(tf.squeeze(x_sorted), tf.squeeze(t_sorted), indexing='ij')
    x_final = tf.reshape(X, (-1, 1))
    t_final = tf.reshape(T, (-1, 1))
    return x_final, t_final

def train_step(model, lr, x, t, a_x, b_x):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x, t, a_x, b_x)
    gradients = tape.gradient(loss, model.trainable_variables)
    lr.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def segmented_train_step(model, seg_lr, x, t, a_x, b_x):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x, t, a_x, b_x)
    gradients = tape.gradient(loss, model.trainable_variables)
    seg_lr.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def segmented_train(model, first_lr, second_lr, xn_pts, tn_pts, epochs, a_x,
                    b_x, final_t, progress, x_train, t_train, segment):
    losses = []
    if x_train is None:
        x, t = find_mesh_subintervals(xn_pts, tn_pts, a_x, b_x, final_t)
    else:
        x, t = x_train, t_train

    for i in range(epochs):
        if i <= segment:
            loss = segmented_train_step(model, first_lr, x, t, a_x, b_x)
        else:
            loss = train_step(model, second_lr, x, t, a_x, b_x)
        
        current_loss = float(loss)
        losses.append(current_loss)
        if i % progress == 0 or i == epochs - 1:
            print(f"Epoch {i}, Loss {current_loss:.6f}")
            
    return losses, x, t

def prepare_and_plot(model, losses, a_x, b_x, final_t, case_name, fd_data=None):
    print(f"\nPreparing plots for {case_name}...")
    
    # 1. Create Directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outs', case_name))
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir) # Clean start
    os.makedirs(os.path.join(base_dir, "log"))
    os.makedirs(os.path.join(base_dir, "plot"))

    # 2. Write parameters.txt for Plotter
    with open(os.path.join(base_dir, "log", "parameters.txt"), "w") as f:
        f.write("Generated by demo script\nParam 2\nProblem : SaintVenant\n")
        
    # 3. Generate Grid for Plotting
    # Use FD resolution if available, otherwise default
    if fd_data:
        x_vals, t_vals, H_fd, U_fd = fd_data
        nx = len(x_vals)
        nt = len(t_vals)
        
        # Transpose FD data to (nt, nx) to match meshgrid 'xy' indexing
        # FD solver returns (nx, nt), we need (nt, nx) for plotting where T is rows, X is cols
        H_fd = H_fd.T
        U_fd = U_fd.T
    else:
        nx, nt = 100, 100
        x_vals = np.linspace(a_x, b_x, nx)
        t_vals = np.linspace(0, final_t, nt)
    
    # Use default indexing='xy'. 
    # X will be (nt, nx), T will be (nt, nx).
    # This corresponds to len(t_vals) rows and len(x_vals) columns.
    X, T = np.meshgrid(x_vals, t_vals) 
    
    # Flatten for PINN prediction (N, 2)
    x_flat = X.flatten()[:, None]
    t_flat = T.flatten()[:, None]
    xt = np.hstack([x_flat, t_flat])
    xt_tf = tf.convert_to_tensor(xt, dtype=tf.float32)
    
    # 4. Predict with PINN
    pred = model(xt_tf).numpy()
    # Swap: [u, h] -> [h, u]
    pred_swapped = pred[:, [1, 0]] 
    
    # 5. Prepare Exact Solution
    if fd_data:
        # Flatten transposed FD arrays to match meshgrid 'xy' flatten order
        h_ex = H_fd.flatten()[:, None]
        u_ex = U_fd.flatten()[:, None]
        exact_vals = np.hstack([h_ex, u_ex])
    else:
        print("Warning: No exact solution provided, using prediction as reference.")
        exact_vals = pred_swapped

    # 6. Prepare Data Structures
    sol_ex = (xt, exact_vals)
    
    z_vals = zb(tf.convert_to_tensor(x_flat, dtype=tf.float32)).numpy()
    par_ex = (xt, z_vals)
    
    functions = {
        'sol_NN': pred_swapped,
        'sol_std': np.zeros_like(pred_swapped), 
        'sol_samples': [pred_swapped],
        'par_NN': z_vals,
        'par_std': np.zeros_like(z_vals),
        'par_samples': [z_vals]
    }
    
    data = {
        "sol_ex": sol_ex,
        "par_ex": par_ex,
        "sol_ns": None,
        "par_ns": None
    }
    
    # 7. Call Plotter
    plotter = Plotter(base_dir)
    
    # Override Plotter scaling
    plotter.scale_x = 1.0
    plotter.scale_t = 1.0
    plotter.label_x = "x (m)"
    plotter.label_t = "t (s)"
    
    plotter.plot_losses(({"Total": losses}, {"Total": losses}))
    plotter.plot_confidence(data, functions)
    plotter.plot_nn_samples(data, functions)
    
    # Add error and QQ plots
    plotter.plot_full_domain_error(data, functions)
    plotter.plot_qq(data, functions)
    
    print(f"Plots saved to {base_dir}/plot")


# --- Example 1: Flat Rectangular Channel ---

print("=== Setting up Example 1 ===")
# Global variables for Ex1
w = 1.0; g = 9.81; rho = 1000; kn = 1.0; xmax = 20.0; nu = 1e-6

def Af(x, h):
    return h * w

def A1(x, h):
    if tf.is_tensor(x):
        return tf.zeros_like(x)
    return 0.0

def A2(x, h):
    if tf.is_tensor(h):
        return tf.ones_like(h) * w
    return w

def Pw(x, h):
    return w + 2.0 * h

def dPw(x, h):
    if tf.is_tensor(h):
        return tf.ones_like(h) * 2.0
    return 2.0

def Dw(x, h):
    return h 

def zb(x):
    if tf.is_tensor(x):
        return tf.ones_like(x) * 0.6
    return 0.6

def dzb(x):
    if tf.is_tensor(x):
        return tf.ones_like(x) * 0.0
    return 0.0

def manning_n():
    # Return simple float for non-TF use, convert in graph if needed
    return 0.015

def Sf(x, h, u):
    # Check if inputs are Tensors
    is_tf = tf.is_tensor(h) or tf.is_tensor(u)
    
    R = Af(x, h) / Pw(x, h)
    n = manning_n()
    
    if is_tf:
        n_tf = tf.convert_to_tensor(n, dtype=tf.float32)
        kn_tf = tf.convert_to_tensor(kn, dtype=tf.float32)
        return (tf.square(u) * tf.square(n_tf)) / (tf.square(kn_tf) * tf.pow(R,4/3))
    else:
        # Numpy/Scalar version
        return (u**2 * n**2) / (kn**2 * R**(4/3))
def h_init(x): return 1.0 - zb(x)
def u_init(x): return 2.0 / h_init(x)
def u_bcleft(t): 
    if tf.is_tensor(t): return u_init(tf.zeros_like(t))
    return u_init(0.0)
def h_bcleft(t): 
    if tf.is_tensor(t): return h_init(tf.zeros_like(t))
    return h_init(0.0)

# Training Ex1
nn = 20; l = 5
xn_pts = 30; tn_pts = 30
a_x = 0.0; b_x = 20.0; final_t = 32
epochs = 20000 # Restored to original
progress = 1000
segment = epochs/7
first_lr = 1e-3; second_lr = 7e-5

x_train, t_train = find_mesh_subintervals(xn_pts, tn_pts, a_x, b_x, final_t)
lr1 = tf.keras.optimizers.Adam(learning_rate=first_lr)
lr2 = tf.keras.optimizers.Adam(learning_rate=second_lr)
model_ex1 = PINN(neurons=nn, layers=l, activation='tanh')
_ = model_ex1(tf.zeros((1, 2))) # Build model

print(f"Starting Training for Example 1 ({epochs} epochs)...")
start = timeit.default_timer()
losses_ex1, _, _ = segmented_train(model_ex1, lr1, lr2, xn_pts, tn_pts, epochs, a_x, b_x,
                                   final_t, progress, x_train, t_train, segment)
print(f"Training time: {timeit.default_timer() - start:.2f}s")

# Generate Exact Solution via Finite Difference
# Use square grid to simplify Plotter reshaping logic (100x100 = 10000 points)
fd_data_ex1 = solve_finite_difference(a_x, b_x, 100, final_t, 100, 
                                      h_init, u_init, h_bcleft, u_bcleft)

prepare_and_plot(model_ex1, losses_ex1, a_x, b_x, final_t, "demo_StVenant_Ex1", fd_data_ex1)


# --- Example 2: Gaussian Bump ---

print("\n=== Setting up Example 2 ===")

# Redefine Global variables for Ex2
def zb(x):
    # Remove Gaussian bump, make it flat
    if tf.is_tensor(x):
        return tf.zeros_like(x)
    return np.zeros_like(x)

def dzb(x):
    # Slope is zero for flat bed
    if tf.is_tensor(x):
        return tf.zeros_like(x)
    return np.zeros_like(x)

def h_init(x): return 0.75 - zb(x) # 0.75 flat depth
def u_init(x): return 7.5 / h_init(x) # 10.0 flat velocity

def h_bcleft(t):
   # Keep flood wave logic
   is_tf = tf.is_tensor(t)
   if is_tf:
       return tf.where(t > 16.0, tf.constant(1.5, dtype=tf.float32), 0.75 - zb(tf.zeros_like(t)))
   else:
       if t > 16.0: return 1.5
       return 0.75 - zb(0.0)

def u_bcleft(t): 
    if tf.is_tensor(t):
        return u_init(tf.zeros_like(t))
    return u_init(0.0)

# Training Ex2
epochs_ex2 = 20000 # Restored to original
progress_ex2 = 1000
segment_ex2 = epochs_ex2/7
first_lr_ex2 = 2e-3; second_lr_ex2 = 1e-4

x_train_ex2, t_train_ex2 = find_mesh_subintervals(40, 40, a_x, b_x, final_t)
lr1_ex2 = tf.keras.optimizers.Adam(learning_rate=first_lr_ex2)
lr2_ex2 = tf.keras.optimizers.Adam(learning_rate=second_lr_ex2)
model_ex2 = PINN(neurons=20, layers=5, activation='tanh')
_ = model_ex2(tf.zeros((1, 2))) # Build model

print(f"Starting Training for Example 2 ({epochs_ex2} epochs)...")
start_ex2 = timeit.default_timer()
losses_ex2, _, _ = segmented_train(model_ex2, lr1_ex2, lr2_ex2, 40, 40, epochs_ex2, a_x, b_x,
                                   final_t, progress_ex2, x_train_ex2, t_train_ex2, segment_ex2)
print(f"Training time: {timeit.default_timer() - start_ex2:.2f}s")

# Generate Exact Solution via Finite Difference
fd_data_ex2 = solve_finite_difference(a_x, b_x, 100, final_t, 100,
                                      h_init, u_init, h_bcleft, u_bcleft)

prepare_and_plot(model_ex2, losses_ex2, a_x, b_x, final_t, "demo_StVenant_Ex2", fd_data_ex2)

print("\nDone.")
