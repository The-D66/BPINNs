import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 配置参数与物理环境
# ==========================================
L = 10.0  # 空间长度 (m)
N = 100  # 空间网格数
dx = L / N
g = 9.81  # 重力加速度
t_max = 2.0  # 模拟时长 (s)
dt_eval = 0.05  # 数据采样间隔 (用于训练和绘图)

# === 改进点 Step 2: 预测时的精细步长 ===
prediction_sub_steps = 5
dt_pred = dt_eval / prediction_sub_steps

# 周期性边界的空间坐标
x = np.linspace(0, L, N, endpoint=False)

# 设备选择
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on device: {device}")


# ==========================================
# 2. 生成 Ground Truth 数据
# ==========================================
def shallow_water_dynamics(y, t, g, dx, N):
  state = y.reshape(2, N)
  h = state[0]
  u = state[1]
  h_x = (np.roll(h, -1) - np.roll(h, 1)) / (2 * dx)
  u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
  dh_dt = -h * u_x - u * h_x
  du_dt = -u * u_x - g * h_x
  return np.concatenate([dh_dt, du_dt])


h0 = 1.0 + 0.1 * np.exp(-(x - L / 4)**2 / 0.5)
u0 = 0.0 * x
y0 = np.concatenate([h0, u0])

t_span = np.arange(0, t_max, dt_eval)
y_true = odeint(shallow_water_dynamics, y0, t_span, args=(g, dx, N))
data_true = y_true.reshape(-1, 2, N)
print(f"数据生成完毕。形状: {data_true.shape}")


# ==========================================
# 3. 定义 HNN
# ==========================================
class HNN_ShallowWater(nn.Module):
  def __init__(self, num_points, dx):
    super().__init__()
    self.N = num_points
    self.dx = dx

    self.energy_net = nn.Sequential(
        nn.Conv1d(2, 64, kernel_size=5, padding=2, padding_mode='circular'),
        nn.Tanh(),
        nn.Conv1d(64, 64, kernel_size=5, padding=2, padding_mode='circular'),
        nn.Tanh(),
        nn.Conv1d(64, 1, kernel_size=5, padding=2, padding_mode='circular')
    )

    self.diff_filter = nn.Conv1d(
        1, 1, kernel_size=3, padding=1, bias=False, padding_mode='circular'
    )
    diff_kernel = torch.tensor([[[1.0, 0.0, -1.0]]]) / (2 * self.dx)
    self.diff_filter.weight.data = diff_kernel
    self.diff_filter.weight.requires_grad = False

  def get_energy(self, x):
    """计算能量 (Energy)"""
    energy_density = self.energy_net(x)
    H = torch.sum(energy_density, dim=2).sum(dim=1)
    return H

  def get_mass(self, x):
    """计算质量 (Mass): Sum(h)"""
    # x shape: [Batch, 2, N], h is channel 0
    h = x[:, 0, :]
    return torch.sum(h, dim=1)

  def forward(self, x):
    with torch.enable_grad():
      x = x.detach().requires_grad_(True)
      energy_density = self.energy_net(x)
      H = torch.sum(energy_density, dim=2).sum()

      is_training = self.training
      grads = torch.autograd.grad(H, x, create_graph=is_training)[0]
      dH_dh = grads[:, 0:1, :]
      dH_du = grads[:, 1:2, :]

    neg_dx_dH_du = self.diff_filter(dH_du)
    neg_dx_dH_dh = self.diff_filter(dH_dh)

    return torch.cat([neg_dx_dH_du, neg_dx_dH_dh], dim=1)


# ==========================================
# 4. 模型训练 (优化策略升级)
# ==========================================
model = HNN_ShallowWater(num_points=N, dx=dx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === 优化策略 1: 使用中心差分生成高质量导数标签 ===
# 原始: (x[t+1] - x[t]) / dt  -> 偏差大
# 改进: (x[t+1] - x[t-1]) / 2dt -> 精度高，更接近瞬时导数
X_full = torch.tensor(data_true, dtype=torch.float32).to(device)

# 只要中间的时间点 (t=1 到 t=T-1)
X_train = X_full[1:-1]

# 对应的导数标签：中心差分
Y_target = (X_full[2:] - X_full[:-2]) / (2 * dt_eval)

print(f"训练集大小: {X_train.shape[0]}")


# 计算真实物理能量与质量用于正则化
def compute_true_energy_torch(x_tensor):
  h = x_tensor[:, 0, :]
  u = x_tensor[:, 1, :]
  kinetic = 0.5 * h * u**2
  potential = 0.5 * g * h**2
  return torch.sum(kinetic + potential, dim=1)


H_true_labels = compute_true_energy_torch(X_train).detach()
M_true_labels = torch.sum(X_train[:, 0, :], dim=1).detach()  # Mass = sum(h)

print("开始训练 HNN (高质量标签 + 质量/能量双正则化)...")
model.train()
start_time = time.time()

for epoch in range(2501):
  optimizer.zero_grad()

  # 1. 动力学 Loss
  pred_derivs = model(X_train)
  loss_dynamics = nn.MSELoss()(pred_derivs, Y_target)

  # 2. 能量正则化 (Weak Constraint)
  H_pred = model.get_energy(X_train)
  loss_energy = nn.MSELoss()(H_pred, H_true_labels)

  # 3. === 优化策略 2: 质量正则化 (Mass Conservation) ===
  M_pred = model.get_mass(X_train)
  loss_mass = nn.MSELoss()(M_pred, M_true_labels)

  # 组合 Loss:
  # Dynamics 主导
  # Energy 辅助标度 (1e-4)
  # Mass 辅助约束 (1e-4) - 帮助稳定波形高度
  loss = loss_dynamics + 1e-4 * loss_energy + 1e-4 * loss_mass

  loss.backward()
  optimizer.step()

  if epoch % 500 == 0:
    print(
        f"Epoch {epoch}, Total: {loss.item():.6f}, Dyn: {loss_dynamics.item():.6f}, Mass: {loss_mass.item():.6f}"
    )

print(f"训练耗时: {time.time() - start_time:.2f}s")

# ==========================================
# 5. 预测推演
# ==========================================
print(f"\n开始隐式辛积分预测 (Sub-steps={prediction_sub_steps})...")


def implicit_midpoint_step(model, state, dt, max_iters=15, tol=1e-6):
  k1 = model(state)
  x_mid = state + 0.5 * dt * k1
  for _ in range(max_iters):
    f_mid = model(x_mid)
    x_mid_new = state + 0.5 * dt * f_mid
    if torch.norm(x_mid_new - x_mid) < tol:
      x_mid = x_mid_new
      break
    x_mid = x_mid_new
  return state + dt * model(x_mid)


# 注意：这里我们从 X_train[0] 开始预测，这实际上是 data_true[1]
# 为了和原始 plot 对齐，我们可以取 X_full[0] 作为起点
pred_states = [X_full[0:1]]
current_state = X_full[0:1]

model.eval()
with torch.no_grad():
  for i in range(len(t_span) - 1):
    for _ in range(prediction_sub_steps):
      current_state = implicit_midpoint_step(model, current_state, dt_pred)
    pred_states.append(current_state)

pred_array = torch.cat(pred_states, dim=0).cpu().numpy()


# ==========================================
# 6. 绘图与分析
# ==========================================
def compute_physics_energy(data):
  h = data[:, 0, :]
  u = data[:, 1, :]
  kinetic = 0.5 * h * u**2
  potential = 0.5 * g * h**2
  return np.sum(kinetic + potential, axis=1)


E_true = compute_physics_energy(data_true)
E_pred = compute_physics_energy(pred_array)

plt.figure(figsize=(10, 5))
plt.plot(t_span, E_true, 'k-', label='Ground Truth', linewidth=2)
plt.plot(
    t_span, E_pred, 'r--', label='HNN (Central Diff + Mass Reg)', linewidth=2
)
mean_E = np.mean(E_true)
plt.ylim(mean_E * 0.95, mean_E * 1.05)
plt.title(f"Total Energy Conservation")
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(15, 5))
plot_indices = [0, 10, 20, 30]
for i, idx in enumerate(plot_indices):
  if idx >= len(t_span):
    break
  plt.subplot(1, 4, i + 1)
  plt.plot(x, data_true[idx, 0, :], 'k-', label='True')
  plt.plot(x, pred_array[idx, 0, :], 'r--', label='HNN')
  plt.title(f"t = {t_span[idx]:.2f}s")
  plt.ylim(0.8, 1.2)
  plt.grid(True, alpha=0.3)
  if i == 0:
    plt.legend()

plt.tight_layout()
plt.show()
