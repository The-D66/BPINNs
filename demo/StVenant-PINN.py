#!/usr/bin/env python
# coding: utf-8

# # First-Order Nonlinear Partial Differential Equation System

# **St. Venant's Equations (圣维南方程)**
# 
# 这些方程描述了明渠中两个因变量的演变：横截面过水面积 $A$ 和沿纵向 $x$ 的流量 $Q(x,t)$，随时间 $t$ 的变化。
# 
# 方程组可表示为：
# \begin{align}
#     & \\\[0.1em]
#     & \displaystyle \frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = 0, \quad (\text{质量守恒/连续性方程}) \\\[1em]
#     & \displaystyle \frac{\partial V}{\partial t} + V\frac{\partial V}{\partial x} + g\frac{\partial y}{\partial x} = g(S_0 - S_f), \quad (\text{动量守恒方程}) \\\[0.2em]
#     &
# \end{align}
# 
# 其中 $y$ 是水深，$V$ 是平均流速，$S_0$ 是河床坡度，$S_f$ 是模拟边界剪切应力能量损失的摩擦坡度。

# ## Import necessary libraries

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import pandas as pd
from IPython.display import display, HTML
import timeit

# ## Neural Network Class (神经网络类)

# In[ ]:


class PINN(tf.keras.Model):
    """
    物理信息神经网络 (PINN) 模型定义
    输入: (x, t) 坐标
    输出: [u, h] (流速, 水深)
    """
    def __init__(self, neurons=16, layers=4, activation='tanh'):
        super(PINN, self).__init__()
        self.hidden_layers = []

        # 创建隐藏层 (Create hidden layers)
        # layers - 1 是因为最后一层是输出层
        for _ in range(layers - 1):
            self.hidden_layers.append(tf.keras.layers.Dense(neurons, activation=activation))

        # 输出层包含2个神经元：一个对应 u(x,t)，一个对应 h(x,t)
        self.output_layer = tf.keras.layers.Dense(2)

    def call(self, x):
        z = x
        for layer in self.hidden_layers:
            z = layer(z)
        return self.output_layer(z)

# ## Derivative function (导数计算函数)

# In[ ]:


def derivative(model, x, t, output_index=0):
    """
    使用 TensorFlow 的自动微分 (AutoDiff) 计算模型输出对输入 (x, t) 的导数。
    
    参数:
        model: PINN 模型实例
        x, t: 输入的空间和时间张量
        output_index: 0 代表计算 u 的导数, 1 代表计算 h 的导数
    
    返回:
        out: 模型输出值 (u 或 h)
        out_x: 对 x 的一阶偏导数 ∂out/∂x
        out_t: 对 t 的一阶偏导数 ∂out/∂t
    """
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, t]) # 监视输入变量以进行微分
        xt = tf.concat([x, t], axis=1) # 拼接 x 和 t 作为网络输入
        uy = model(xt) # 前向传播
        out = uy[:, output_index:output_index+1] # 提取对应的输出 (u 或 h)
        
        # 针对水深 h (index 1) 的物理约束技巧：
        # 如果 h < 0，使用 Softplus 激活函数平滑处理，避免负水深导致的数值不稳定
        if output_index == 1:
            eps = 1e-1
            out = tf.where(out < 0.0, tf.nn.softplus(out) * eps, out)
            
    out_x = tape1.gradient(out, x) # 计算 ∂/∂x
    out_t = tape1.gradient(out, t) # 计算 ∂/∂t
    del tape1
    return out, out_x, out_t

# ## Residual Function (残差函数)

# In[ ]:


def residual(model, x, t):
    """
    计算圣维南方程组的物理残差 (Physics Residuals)。
    残差越接近 0，说明模型的预测越符合物理规律。
    """
    # 获取 u 和 h 及其导数
    u, u_x, u_t = derivative(model, x, t, 0) # Derivatives of u / velocity
    h, h_x, h_t = derivative(model, x, t, 1) # Derivatives of h / water depth

    # 1. 连续性方程 (Continuity equation): 
    # A2*∂h/∂t + u*(A1 + A2*∂h/∂x) + A*∂u/∂x = 0
    # 这由 ∂A/∂t + ∂(AV)/∂x = 0 推导而来
    continuity_residual = A2(x,h)*h_t + u*(A1(x,h) + A2(x,h)*h_x) + Af(x,h)*u_x

    # 2. 动量方程 (Momentum equation): 
    # ∂u/∂t + u*∂u/∂x + g*∂h/∂x + g*(Sf - S0) = 0
    momentum_residual = u_t + u*u_x + g*h_x + g*(Sf(x,h,u) - dzb(x))

    return continuity_residual, momentum_residual

# ## Loss function (损失函数)

# In[ ]:


def loss_fn(model, x, t, a_x, b_x):
    """
    总损失函数 = 物理残差损失 + 初始条件损失 + 边界条件损失
    """
    # 1. 物理残差损失 (Residual Loss)
    continuity_residual, momentum_residual = residual(model, x, t)
    mse_continuity = tf.reduce_mean(tf.square(continuity_residual))
    mse_momentum = tf.reduce_mean(tf.square(momentum_residual))

    # 2. 初始条件损失 (Initial condition, t=0)
    t0 = tf.zeros_like(x)
    xt0 = tf.concat([x, t0], axis=1)
    
    # 预测 t=0 时的值
    u0_pred = model(xt0)[:, 0:1]
    h0_pred = model(xt0)[:, 1:2]
    
    # 真实初始值
    u0_true = u_init(x)
    h0_true = h_init(x)

    mse_ic_u = tf.reduce_mean(tf.square(u0_pred - u0_true))
    mse_ic_h = tf.reduce_mean(tf.square(h0_pred - h0_true))

    # 3. 边界条件损失 (Boundary condition, x=a_x 左边界)
    xl = tf.ones_like(t) * a_x
    xlt = tf.concat([xl, t], axis=1)
    
    # 预测左边界的值
    u_l_pred = model(xlt)[:, 0:1]
    h_l_pred = model(xlt)[:, 1:2]
    
    # 真实左边界值
    u_l_true = u_bcleft(t)
    h_l_true = h_bcleft(t)

    mse_bc_u = tf.reduce_mean(tf.square(u_l_pred - u_l_true))
    mse_bc_h = tf.reduce_mean(tf.square(h_l_pred - h_l_true))

    # 总损失求和
    total_loss = mse_continuity + mse_momentum + mse_ic_u + mse_ic_h + mse_bc_u + mse_bc_h
    return total_loss

# ## Subinterval Sampling (子区间采样)

# 域被划分为 `n_pts` 个子区间（空间和时间）。每个子区间内随机采样一个点，然后组合成网格。
# 这种随机采样策略有助于避免神经网络在训练时过拟合固定网格点。

# In[ ]:


def find_mesh_subintervals(xn_pts, tn_pts, a_x, b_x, final_t):
    # 确定子区间的边缘
    x_edges = tf.linspace(a_x, b_x, xn_pts + 1)
    t_edges = tf.linspace(0.0, final_t, tn_pts + 1)

    # 在每个子区间内产生随机偏移
    x_offsets = tf.random.uniform((xn_pts,), 0, 1, dtype=tf.float32)
    t_offsets = tf.random.uniform((tn_pts,), 0, 1, dtype=tf.float32)

    # 计算实际采样点坐标
    x_points = x_edges[:-1] + x_offsets * (x_edges[1:] - x_edges[:-1])
    t_points = t_edges[:-1] + t_offsets * (t_edges[1:] - t_edges[:-1])

    x_sorted = tf.sort(x_points, axis=0)
    t_sorted = tf.sort(t_points, axis=0)

    # 创建网格 (Meshgrid)
    X, T = tf.meshgrid(tf.squeeze(x_sorted), tf.squeeze(t_sorted), indexing='ij')
    x_final = tf.reshape(X, (-1, 1))
    t_final = tf.reshape(T, (-1, 1))

    return x_final, t_final

# ## Segmented Train Step (分段训练步骤)

# In[ ]:


@tf.function
def train_step(model, lr, x, t, a_x, b_x):
    """标准训练步"""
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x, t, a_x, b_x)
    gradients = tape.gradient(loss, model.trainable_variables)
    lr.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def segmented_train_step(model, seg_lr, x, t, a_x, b_x):
    """使用不同学习率的训练步"""
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x, t, a_x, b_x)
    gradients = tape.gradient(loss, model.trainable_variables)
    seg_lr.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# ## Segmented Train Loop (分段训练循环)

# In[ ]:


def segmented_train(model, first_lr, second_lr, xn_pts, tn_pts, epochs, a_x,
                    b_x, final_t, progress, x_train, t_train, segment):
    """
    训练循环，支持两阶段学习率策略。
    前 `segment` 个 epoch 使用 `first_lr`，之后使用 `second_lr`。
    """
    losses = []
    if x_train is None:
        # 如果未提供训练点，则生成新的采样点
        x, t = find_mesh_subintervals(xn_pts, tn_pts, a_x, b_x, final_t)
    else:
        x, t = x_train, t_train

    for i in range(epochs):
        if i <= segment:
            loss = segmented_train_step(model, first_lr, x, t, a_x, b_x)
        else:
            loss = train_step(model, second_lr, x, t, a_x, b_x)

        losses.append(float(loss))
        if i % progress == 0:
            print(f"Epoch {i}, Loss {float(loss)}")
    return losses, x, t

# ## Example 1: Flat Rectangular Channel (案例1：平底矩形渠道)

# In[ ]:


# 物理和几何常数
w = 1.0    # 渠道宽度 [m]
g = 9.81   # 重力 [m/s²]
rho = 1000 # 密度
kn = 1.0   # 曼宁转换系数

xmax = 20.0  # 渠道长度 [m]
nu = 1e-6    # 运动粘度

# 几何相关函数 (Geometry Dependent Functions)

def Af(x, h):
    """过水断面面积"""
    return h * w

def A1(x, h):
    """∂A/∂x (矩形渠道为0)"""
    return tf.zeros_like(x)

def A2(x, h):
    """∂A/∂h (矩形渠道为宽度 w)"""
    return tf.ones_like(h) * w

def Pw(x, h):
    """湿周"""
    return w + 2.0 * h

def dPw(x, h):
    """∂P/∂h"""
    return tf.ones_like(h) * 2.0

def Dw(x, h):
    """水力深度 D = A/T"""
    return h 

def zb(x):
  """河床高程 (常数 0.6)"""
  return tf.ones_like(x) * 0.6

def dzb(x):
  """河床坡度 (0)"""
  return tf.ones_like(x) * 0.0

# 模型组件 (Model Components)

def manning_n():
    """曼宁系数"""
    return tf.constant(0.015, dtype=tf.float32)

def Sf(x, h, u):
    """摩擦坡度 (Manning's formula)"""
    R = Af(x, h) / Pw(x, h)  # 水力半径
    n = manning_n()
    return (tf.square(u) * tf.square(n)) / (tf.square(kn) * tf.pow(R,4/3))

def celerity(x, h):
    """波速"""
    return tf.sqrt(g * Dw(x, h))

def Froude(x, h, u):
    """弗劳德数"""
    return u / celerity(x, h)

def Reynolds(x, h, u):
    """雷诺数"""
    return u / nu * Af(x, h) / Pw(x, h)

# 边界条件函数 (Boundary Conditions)

def h_init(x):
    """初始水深"""
    return 1.0 - zb(x)

def u_init(x):
    """初始流速"""
    return 2.0 / h_init(x)

def u_bcleft(t):
    """左侧流速边界"""
    return u_init(tf.zeros_like(t))

def h_bcleft(t):
    """左侧水深边界"""
    return h_init(tf.zeros_like(t))

# 检查 Froude 数和 Reynolds 数
xp=0.5*xmax
print("Example 1 Check:")
print(Froude(xp,h_init(xp),u_init(xp)))
# ...

# ### Training Setup (训练设置)

# In[ ]:


nn = 20 # 神经元数量
l = 5   # 层数

xn_pts = 30 # 空间采样点数
tn_pts = 30 # 时间采样点数
a_x = 0.0 # 左边界
b_x = 20.0 # 右边界
final_t = 32 # 总时间

epochs = 20000 # 训练轮数
progress = 10000 # 打印频率
segment = epochs/7 # 学习率切换点

first_lr = 1e-3 # 初始学习率
second_lr = 7e-5 # 后期学习率

# 生成训练点
x_train_ex1, t_train_ex1 = find_mesh_subintervals(xn_pts, tn_pts, a_x, b_x, final_t)

# 定义优化器
lr1 = tf.keras.optimizers.Adam(learning_rate=first_lr)
lr2 = tf.keras.optimizers.Adam(learning_rate=second_lr)

# 编译模型
model_ex1 = PINN(neurons=nn, layers=l, activation='tanh')

start = timeit.default_timer()
# 开始训练
print("Starting Training for Example 1...")
losses_ex1, xpoints_ex1, tpoints_ex1 = segmented_train(model_ex1, lr1, lr2, xn_pts, tn_pts, epochs, a_x, b_x,
                                                    final_t, progress, x_train_ex1, t_train_ex1, segment)
end = timeit.default_timer()
print("\nTraining time:", end - start, "seconds")

# ... (Plotting code for Loss, Velocity, Water Surface, Discharge) ...
# ... (Same logic as Finite Difference plotting, but using model predictions) ...


# ## Example 2: Gaussian Bump (案例2：高斯凸起河床)

# 重定义参数和几何函数

# In[ ]:


w = 1.0    
g = 9.81   
rho = 1000 
kn = 1.0   
xmax = 20.0 
nu = 1e-6    

def Af(x, h): return h * w
def A1(x, h): return tf.zeros_like(x)
def A2(x, h): return tf.ones_like(h) * w
def Pw(x, h): return w + 2.0 * h
def dPw(x, h): return tf.ones_like(h) * 2.0
def Dw(x, h): return h 

def zb(x):
    """高斯凸起河床"""
    ampl = 0.25
    mean = xmax / 2
    stdev = xmax / 20
    return ampl * tf.exp(-0.5 * ((x - mean)/stdev)**2)

def dzb(x):
    """高斯凸起的坡度"""
    ampl = 0.25
    mean = xmax / 2
    stdev = xmax / 20
    return -0.5 * 2 * (x - mean)/stdev * ampl * tf.exp(-0.5 * ((x - mean)/stdev)**2)

def manning_n(): return tf.constant(0.015, dtype=tf.float32)

def Sf(x, h, u):
    R = Af(x, h) / Pw(x, h)
    n = manning_n()
    return (tf.square(u) * tf.square(n)) / (tf.square(kn) * tf.pow(R,4/3))

def celerity(x, h): return tf.sqrt(g * Dw(x, h))
def Froude(x, h, u): return u / celerity(x, h)
def Reynolds(x, h, u): return u / nu * Af(x, h) / Pw(x, h)

# 案例2的边界条件
def h_init(x):
    return 0.75 - zb(x)

def u_init(x):
    return 7.5 / h_init(x)

def h_bcleft(t):
   """
   时间相关的左侧边界条件
   t > 16s 时模拟洪水波
   """
   return tf.where(t > 16.0,
                  tf.constant(1.5, dtype=tf.float32),
                  0.75 - zb(tf.zeros_like(t)))

def u_bcleft(t):
    return u_init(tf.zeros_like(t))


# 案例2 训练设置
nn_ex2 = 20
l_ex2 = 5
xn_pts_ex2 = 40
tn_pts_ex2 = 40
a_x_ex2 = 0.0
b_x_ex2 = 20.0
final_t_ex2 = 32

epochs_ex2 = 20000
progress_ex2 = 10000
segment_ex2 = epochs_ex2/7

first_lr_ex2 = 2e-3
second_lr_ex2 = 1e-4

# 生成训练点 (Example 2)
# 注意：此处注释掉的代码如果开启，将使用子区间采样，否则可能需要重新实现 find_mesh_subintervals 的调用
x_train_ex2, t_train_ex2 = find_mesh_subintervals(xn_pts_ex2, tn_pts_ex2, a_x_ex2, b_x_ex2, final_t_ex2)

lr1_ex2 = tf.keras.optimizers.Adam(learning_rate=first_lr_ex2)
lr2_ex2 = tf.keras.optimizers.Adam(learning_rate=second_lr_ex2)

model_ex2 = PINN(neurons=nn_ex2, layers=l_ex2, activation='tanh')

start_ex2 = timeit.default_timer()
print("Starting Training for Example 2...")
losses_ex2, xpoints_ex2, tpoints_ex2 = segmented_train(model_ex2, lr1_ex2, lr2_ex2, xn_pts_ex2,
                                                       tn_pts_ex2, epochs_ex2, a_x_ex2, b_x_ex2,
                                                       final_t_ex2, progress_ex2, x_train_ex2,
                                                       t_train_ex2, segment_ex2)
end_ex2 = timeit.default_timer()
print("\nTraining time:", end_ex2 - start_ex2, "seconds")

# ... (Plotting code for Example 2) ...