import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import optimize
from matplotlib import pyplot as plt
from IPython.display import display, HTML

# ==========================================
# 有限差分系统定义 (Finite Difference Schemes)
# ==========================================
# 这里定义了圣维南方程组的不同离散化格式。
# 核心方程：
# 1. 质量守恒 (Mass): dA/dt + dQ/dx = 0
# 2. 动量守恒 (Momentum): du/dt + u*du/dx + g*dh/dx + g*(Sf - S0) = 0

def stvenant_system_impl(sol,sol_past,sol_left,x,dx,t,dt):
  """
  全隐式格式 (Implicit Scheme)
  sol: 当前待求解的时间步解 [h, u]
  sol_past: 上一时间步解
  sol_left: 空间左侧节点的解
  """
  h = sol[0]; u=sol[1];
  h_past = sol_past[0]; u_past = sol_past[1];
  h_left = sol_left[0]; u_left = sol_left[1];
  
  # 时间导数 (Backward Difference in Time)
  dhdt = (h-h_past)/dt; dudt = (u-u_past)/dt;
  # 空间导数 (Backward Difference in Space)
  dhdx = (h-h_left)/dx; dudx = (u-u_left)/dx;
  
  # 质量方程残差
  mass = A2(x,h)*dhdt + u*(A1(x,h)+A2(x,h)*dhdx) + Af(x,h)*dudx
  # 动量方程残差
  momentum = dudt + u*dudx + g*dhdx + g*(Sf(x,h,u)-dzb(x))
  return np.array([mass,momentum])

def stvenant_system_expl_left(sol,sol_past,sol_left,x_left,dx,t_past,dt):
  """
  左侧显式格式 (Explicit Left Scheme)
  使用左侧节点的已知值来近似空间导数系数
  """
  h = sol[0]; u=sol[1];
  h_past = sol_past[0]; u_past = sol_past[1];
  h_left = sol_left[0]; u_left = sol_left[1];
  
  dhdt = (h-h_past)/dt; dudt = (u-u_past)/dt;
  dhdx = (h-h_left)/dx; dudx = (u-u_left)/dx;
  
  # 系数使用 (x_left, h_left) 计算
  mass = A2(x_left,h_left)*dhdt                         \
       + u*(A1(x_left,h_left) + A2(x_left,h_left)*dhdx) \
       + Af(x_left,h_left)*dudx
  momentum = dudt + u_left*dudx + g*dhdx                \
           + g*(Sf(x_left,h_left,u_left)-dzb(x_left))
  return np.array([mass,momentum])

def stvenant_system_midp_left(sol,sol_past,sol_left,x_left,dx,t_past,dt):
  """
  左侧中点格式 (Midpoint Left Scheme) - 类似于 Box Scheme
  在空间区间的中点进行近似，精度通常比一阶显式更高。
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

def stvenant_system_expl_past(sol,sol_past,sol_left,x_left,dx,t_past,dt):
  """
  过去时间步显式格式 (Explicit Past Scheme)
  使用上一时间步 (past) 的值来近似系数
  """
  h = sol[0]; u=sol[1];
  h_past = sol_past[0]; u_past = sol_past[1];
  h_left = sol_left[0]; u_left = sol_left[1];
  dhdt = (h-h_past)/dt; dudt = (u-u_past)/dt;
  dhdx = (h-h_left)/dx; dudx = (u-u_left)/dx;
  
  mass = A2(x,h_past)*dhdt \
       + u*(A1(x,h_past)+ A2(x,h_past)*dhdx) \
       + Af(x,h_past)*dudx
  momentum = dudt + u_past*dudx + g*dhdx     \
           + g*(Sf(x,h_past,u_past)-dzb(x))
  return np.array([mass,momentum])


def stvenant_system_time_rates(sol,sol_past,sol_left,x,dx,t,dt):
  """
  计算时间变化率 dh/dt 和 du/dt，用于 ODE 求解器
  """
  h = sol[0]; u=sol[1];
  h_past = sol_past[0]; u_past = sol_past[1];
  h_left = sol_left[0]; u_left = sol_left[1];
  dhdx = (h-h_left)/dx; dudx = (u-u_left)/dx;
  # just use the forms dhdt = ... , dudt= ...
  dhdt = - 1/A2(x,h)*(u*(A1(x,h)+A2(x,h)*dhdx) + Af(x,h)*dudx)
  dudt = -(u*dudx + g*dhdx + g*(Sf(x,h,u)-dzb(x)))
  return np.array([dhdt,dudt])

# ==========================================
# 案例 1：平底矩形渠道 (Example 1: Flat Rectangular Channel)
# ==========================================

# 物理参数
w = 1.0    # 渠道宽度 [m]
g = 9.81   # 重力加速度 [m/s²]
rho = 1000 # 水密度 [kg/m^3]
kn = 1.0   # 曼宁系数的单位转换因子

xmax = 20.0  # 渠道总长度 [m]
nu = 1e-6    # 运动粘度 [m^2/s]

# 几何与物理辅助函数
def Af(x, h):
    """过水断面面积 A = h * w"""
    return h * w

def A1(x, h):
    """dA/dx (对于矩形渠道，宽度不变，为0)"""
    return 0.0

def A2(x, h):
    """dA/dh (对于矩形渠道，即宽度 w)"""
    return w

def Pw(x, h):
    """湿周 (Wetted perimeter) = w + 2h"""
    return w + 2.0 * h

def dPw(x, h):
    """dPw/dh"""
    return 2.0

def Dw(x, h):
    """水力深度 (Hydraulic depth) D = A / Top_Width = h"""
    return h 

def zb(x):
  """河床高程 (Bed level) - 案例1为平底"""
  return 0.6

def dzb(x):
  """河床坡度 -dzb/dx - 案例1为0"""
  return 0.0

def manning_n():
    """曼宁糙率系数"""
    return 0.015

def Sf(x, h, u):
    """摩擦坡度 (Friction slope)，使用曼宁公式计算"""
    R = Af(x, h) / Pw(x, h)  # 水力半径 R
    n = manning_n()
    # Manning formula: S_f = (n^2 * u^2) / (R^(4/3))
    return ( u**2 * n**2 )/( kn**2 * R**(4/3) )

def celerity(x, h):
    """波速 c = sqrt(g * D)"""
    return (g*Dw(x,h))**(1/2)

def Froude(x, h, u):
    """弗劳德数 Fr = u / c"""
    return u / celerity(x, h)

def Reynolds(x, h, u):
    """雷诺数 Re = u * R / nu"""
    return u / nu * Af(x, h) / Pw(x, h)

# 初始条件与边界条件
def h_init(x):
    """初始水深"""
    return 1.0 - zb(x)

def u_init(x):
    """初始流速"""
    return 2.0 / h_init(x)

def u_bcleft(t):
    """左侧边界流速 (随时间变化)"""
    return u_init(0)

def h_bcleft(t):
    """左侧边界水深 (随时间变化)"""
    return h_init(0)

# 打印检查一些关键物理量
xp=0.5*xmax
print("Check Froude numbers (Example 1):")
print(Froude(xp,h_init(xp),u_init(xp)))
print(Froude(xp,h_bcleft(xp),u_bcleft(xp)))
print(Froude(xp,h_init(xp),u_bcleft(xp)))
print(Froude(xp,h_bcleft(xp),u_init(xp)))

xp=0.4*xmax
print("Check Reynolds numbers (Example 1):")
print(Reynolds(xp,h_init(xp),u_init(xp)))
# ...

# 生成网格 (Grid Generation)
xsteps = 80; xmax = 20.0;
tsteps = 128; tmax = 0.25*tsteps;
x = np.linspace(0,xmax,xsteps+1)
t = np.linspace(0,tmax,tsteps+1)
dx = x[1]-x[0]
dt = t[1]-t[0]
X,T = np.meshgrid(t,x) # T对应行(x), X对应列(t)
H = np.zeros_like(X)
U = np.zeros_like(X)

# 初始化解数组
for i in range(len(x)):
  H[i,0] = h_init(x[i]) # t=0 时刻所有x的值
  U[i,0] = u_init(x[i])
for j in range(1,len(t)):
  H[0,j] = h_bcleft(t[j]) # x=0 位置所有t的值
  U[0,j] = u_bcleft(t[j])

# 求解循环 (Solver Loop)
# 使用嵌套循环进行时间推进和空间扫描
for j in range(1,len(t)):      # 时间步循环
  for i in range(1,len(x)):    # 空间步循环
    # 获取已知信息
    sol_past = np.array([H[i,j-1],U[i,j-1]]) # 同一位置上一时刻
    sol_left = np.array([H[i-1,j],U[i-1,j]]) # 同一时刻左侧位置 (假设波从左向右传，信息已知)
    sol_guess = sol_past # 初始猜测值
    
    # 使用 scipy.optimize.root 求解非线性方程组
    # 这里使用的是中点格式 (midp)
    output = optimize.root(stvenant_system_midp_left,sol_guess,args=(sol_past,sol_left,x[i-1],dx,t[j-1],dt),
                           method='hybr' , tol=1e-9)

    if not output.success:
      print(" i=",i," j=",j," Froude=",Froude(x[i],sol_left[0],sol_left[1])," error message =",output.message)

    # 更新解矩阵
    H[i,j] = output.x[0]
    U[i,j] = output.x[1]

# 计算河床和水面高程用于绘图
Zb = zb(x) * np.ones_like(X) 
water_surface = H + Zb 

# 动画制作 (Animation)
freq = int(len(t) / 100) if len(t) > 100 else 1

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xlabel('Distance along channel (m)')
ax1.set_ylabel('Velocity (m/s)')
ax1.set_title('Velocity Profile Evolution')
ax1.grid(True)

line_u, = ax1.plot(x, U[:,0], '-g', label=f'Velocity at t = {float(t[0]):.1f} s')
ax1.set_ylim(np.min(U)*0.9, np.max(U)*1.1)
ax1.legend(loc='upper right')

def animate_velocity(i):
    line_u.set_ydata(U[:, i])
    line_u.set_label(f'Velocity at t = {t[i]:.1f} s')
    ax1.legend(loc='upper right')
    return line_u,

u_frames = range(0,len(t), freq)
ani_u = FuncAnimation(fig1, animate_velocity, frames=u_frames, interval=50, repeat=False)
plt.close()
HTML(ani_u.to_html5_video())

# 静态切片图 (Static Snapshots)
snapshot_indices = [0, 4, 8, 16, 32, 48, 128]

plt.figure(figsize=(8,5))
for idx in snapshot_indices:
    if idx < U.shape[1]:
        plt.plot(x, U[:, idx], '--', label=f't = {t[idx]:.1f} s')
plt.xlabel('Distance along channel (m)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Profiles at Different Times')
plt.ylim(np.min(U)*0.9, np.max(U)*1.1)
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1.02,1))
plt.show()

# ... (Water surface animation code omitted for brevity, similar to Velocity)
# ... (Discharge animation code omitted) ...

# 计算流量 Q = U * H
Q = U * H
# ... (Discharge plotting) ...

# 最终时刻状态图
time = -1
u_final = U[:, time]
zb_vals = Zb[:, time] 
water_surface_final = water_surface[:, time] 

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, u_final, linewidth=2)
plt.title(f'Velocity Profile at t = {float(t[time]):.1f} s')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, zb_vals, 'k-', linewidth=2, label='Channel Bed')
plt.plot(x, water_surface_final, 'b-', linewidth=2, label='Water Surface')
plt.fill_between(x, zb_vals, water_surface_final, alpha=0.25)
plt.title(f'Water Surface at t = {float(t[time]):.1f} s')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================================
# 案例 2：带高斯凸起的矩形渠道 (Example 2: Gaussian Bump)
# ==========================================

# 重置参数 (Reset Parameters)
w = 1.0    
g = 9.81   
rho = 1000 
kn = 1.0   
xmax = 20.0 
nu = 1e-6   

# 重新定义几何函数 (Redefine Geometry Functions)
# ... (Af, A1, A2, Pw, dPw, Dw Same as before) ...
def Af(x, h): return h * w
def A1(x, h): return 0.0
def A2(x, h): return w
def Pw(x, h): return w + 2.0 * h
def dPw(x, h): return 2.0
def Dw(x, h): return h 

def zb(x):
    """河床高程：高斯分布形成的凸起 (Gaussian Bump)"""
    ampl = 0.25 # 凸起高度
    mean = xmax / 2 # 中心位置
    stdev = xmax / 20 # 宽度标准差
    return ampl*np.exp(-0.5*((x-mean)/stdev)**2)

def dzb(x):
    """河床坡度：高斯函数的导数"""
    ampl = 0.25
    mean = xmax / 2
    stdev = xmax / 20
    return -0.5*2*(x-mean)/stdev*ampl*np.exp(-0.5*((x-mean)/stdev)**2)

def manning_n(): return 0.015

def Sf(x, h, u):
    R = Af(x, h) / Pw(x, h)
    n = manning_n()
    return ( u**2 * n**2 )/( kn**2 * R**(4/3) )

def celerity(x, h): return (g*Dw(x,h))**(1/2)
def Froude(x, h, u): return u / celerity(x, h)
def Reynolds(x, h, u): return u / nu * Af(x, h) / Pw(x, h)

# 新的初始/边界条件
def u_init(x):
    """初始流速"""
    return 7.5 / h_init(x)

def h_init(x):
    """初始水深"""
    return 0.75 - zb(x)

def u_bcleft(t):
    """左侧流速边界"""
    return u_init(0)

def h_bcleft(t):
  """
  左侧水深边界 (时间相关)
  模拟洪水波：t > 16s 时水深突然增加
  """
  if t>16.0:
    return 1.5
  else:
    return 0.75 - zb(0)

# 检查参数
xp=0.5*xmax
print("Check Froude numbers (Example 2):")
print(Froude(xp,h_init(xp),u_init(xp)))

# 生成网格 - 案例 2
xsteps_ex2 = 80; xmax_ex2 = 20.0;
tsteps_ex2 = 128; tmax_ex2 = 0.25*tsteps_ex2;
x_ex2 = np.linspace(0,xmax_ex2,xsteps_ex2+1)
t_ex2 = np.linspace(0,tmax_ex2,tsteps_ex2+1)
dx_ex2 = x_ex2[1]-x_ex2[0]
dt_ex2 = t_ex2[1]-t_ex2[0]
X_ex2,T_ex2 = np.meshgrid(t_ex2,x_ex2) 
H_ex2 = np.zeros_like(X_ex2)
U_ex2 = np.zeros_like(X_ex2)

# 初始化
for i in range(len(x_ex2)):
  H_ex2[i,0] = h_init(x_ex2[i])
  U_ex2[i,0] = u_init(x_ex2[i])
for j in range(1,len(t_ex2)):
  H_ex2[0,j] = h_bcleft(t_ex2[j])
  U_ex2[0,j] = u_bcleft(t_ex2[j])

# 求解循环 - 案例 2
for j in range(1,len(t_ex2)):
  for i in range(1,len(x_ex2)):
    sol_past_ex2 = np.array([H_ex2[i,j-1],U_ex2[i,j-1]])
    sol_left_ex2 = np.array([H_ex2[i-1,j],U_ex2[i-1,j]])
    sol_guess_ex2 = sol_past_ex2
    
    output_ex2 = optimize.root(stvenant_system_midp_left,sol_guess_ex2,args=(sol_past_ex2,sol_left_ex2,x_ex2[i-1],dx_ex2,
                                                                             t_ex2[j-1],dt_ex2),method='hybr', tol=1e-9)

    if not output_ex2.success:
      print("Error in Ex2: i=",i," j=",j," message =",output_ex2.message)

    H_ex2[i,j] = output_ex2.x[0]
    U_ex2[i,j] = output_ex2.x[1]

# 动画与绘图 - 案例 2
freq_ex2 = int(len(t_ex2) / 100) if len(t_ex2) > 100 else 1
# ... (Animation code similar to Ex1) ...

# 最终时刻分析
time = -1
u_final_ex2 = U_ex2[:, time]  
zb_vals_ex2 = zb(x_ex2) # Recompute bed
water_surface_final_ex2 = H_ex2[:, time] + zb_vals_ex2

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x_ex2, u_final_ex2, linewidth=2)
plt.title(f'Velocity at t = {float(t_ex2[time]):.1f} s')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_ex2, zb_vals_ex2, 'k-', linewidth=2, label='Channel Bed')
plt.plot(x_ex2, water_surface_final_ex2, 'b-', linewidth=2, label='Water Surface')
plt.fill_between(x_ex2, zb_vals_ex2, water_surface_final_ex2, alpha=0.25)
plt.title(f'Water Surface at t = {float(t_ex2[time]):.1f} s')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()