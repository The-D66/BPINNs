# 圣维南方程（浅水方程）的哈密顿形式推导

## 1. 简介

圣维南方程（Saint-Venant Equations），又称浅水方程（Shallow Water Equations, SWE），描述了具有自由表面的流体在重力作用下的流动。对于一维情况，其守恒形式通常表示为：

$$
\begin{aligned}
\frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} &= 0 \\
\frac{\partial (hu)}{\partial t} + \frac{\partial}{\partial x}\left(hu^2 + \frac{1}{2}gh^2\right) &= -gh \frac{\partial z_b}{\partial x} - g \frac{n^2 u |u|}{h^{1/3}}
\end{aligned}
$$

其中：
- $h(x,t)$：水深
- $u(x,t)$：平均流速
- $g$：重力加速度
- $z_b(x)$：河床高程（$S_0 = - \partial z_b / \partial x$）
- $n$：曼宁摩擦系数

## 2. 哈密顿体系结构

为了推导哈密顿形式，我们暂时忽略摩擦项（$n=0$），因为耗散系统通常不仅由哈密顿量描述（需要添加耗散括号）。我们关注理想流体的保守部分。

### 2.1 状态变量

引入守恒变量：
- 质量密度（即水深，假设流体密度 $ho=1$）：$ho = h$
- 动量密度：$m = hu$

### 2.2 哈密顿量（总能量）

系统的总能量 $H$ 由动能和势能组成：

$$ 
\mathcal{H} = \mathcal{K} + \mathcal{P} 
$$

动能密度为 $rac{1}{2} h u^2 = rac{1}{2} rac{m^2}{h}$。
势能密度（以河床 $z_b$ 为基准）为 $rac{1}{2} g h^2 + g h z_b$。

于是，哈密顿泛函为：

$$ H[h, m] = \int_{\Omega} \left( \frac{m^2}{2h} + \frac{1}{2} g h^2 + g h z_b \right) dx 
$$

### 2.3 变分导数

计算哈密顿量对状态变量的变分导数：

$$ 
\frac{\delta H}{\delta h} = -\frac{m^2}{2h^2} + g h + g z_b = -\frac{1}{2}u^2 + g(h + z_b) 
$$

$$ 
\frac{\delta H}{\delta m} = \frac{m}{h} = u 
$$

注意：$rac{\delta H}{\delta h}$ 实际上对应于总压头（Total Head）或伯努利项（Bernoulli term）。

### 2.4 哈密顿演化方程

一维浅水方程可以写成非正则哈密顿形式（Lie-Poisson bracket）：

$$ 
\frac{\partial}{\partial t} \begin{pmatrix} h \\ m \end{pmatrix} = \mathcal{J} \begin{pmatrix} \frac{\delta H}{\delta h} \\ \frac{\delta H}{\delta m} \end{pmatrix} 
$$

其中斜对称算子 $\mathcal{J}$ 为：

$$ 
\mathcal{J} = \begin{pmatrix} 0 & -\partial_x \\ -\partial_x & -\left(\partial_x m + m \partial_x \right) \end{pmatrix} 
$$

或者更常见的形式，质量守恒和动量守恒直接由下式给出：

**质量守恒：**
$$ 
\frac{\partial h}{\partial t} = - \frac{\partial}{\partial x} \left( \frac{\delta H}{\delta m} \right) = - \frac{\partial u}{\partial x} \quad \text{(注意：此处标准形式应为 } -\partial_x m \text{，需修正算子)} 
$$

为了匹配标准的浅水方程：
$$ 
\begin{aligned}
h_t &= - \partial_x m \\
m_t &= - \partial_x \left( \frac{m^2}{h} + \frac{1}{2}gh^2 \right) - gh \partial_x z_b
\end{aligned}
$$ 

我们可以采用 **Gardner-Faddeev-Zakharov** 括号或简单的辛结构形式。对于变量 $(h, \phi)$ （其中 $\phi$ 是速度势，$u = \partial_x \phi$），结构更简单（正则哈密顿系统）：

$$ 
\frac{\partial h}{\partial t} = \frac{\delta H}{\delta \phi}, \quad \frac{\partial \phi}{\partial t} = -\frac{\delta H}{\delta h} 
$$

但在工程中我们使用 $(h, u)$ 或 $(h, m)$。

在 PINN 实现中，我们直接利用守恒律和能量形式。我们可以定义“哈密顿残差”作为满足能量守恒或辛结构的约束，但最直接的物理信息神经网络（PINN）实现通常是求解上述偏微分方程。

如果我们想强调“哈密顿形式”的实现，我们可以显式地构建残差，使其包含变分导数的计算。

### 2.5 包含耗散的公式

对于实际的圣维南方程（含摩擦）：

$$ 
\frac{\partial}{\partial t} \begin{pmatrix} h \\ m \end{pmatrix} = \mathcal{J} \begin{pmatrix} \frac{\delta H}{\delta h} \\ \frac{\delta H}{\delta m} \end{pmatrix} + \begin{pmatrix} 0 \\ -D \end{pmatrix} 
$$

其中耗散项 $D = g \frac{n^2 u |u|}{h^{1/3}} h = g n^2 h^{2/3} u |u|$。

## 3. 神经网络实现形式

我们将实现一个名为 `SaintVenantHamiltonian` 的类。

与直接计算 PDE 残差不同，我们将：
1. 定义哈密顿量 $H(h, u)$。
2. 使用自动微分计算变分导数 $\delta H / \delta h$ 和 $\delta H / \delta u$。
3. 构建演化方程残差。

这将允许网络学习潜在的能量结构，或者至少在物理约束中显式包含能量项。

方程组残差：
1. **质量守恒**：$R_1 = h_t + \partial_x (hu)$
2. **动量守恒（哈密顿形式）**：$R_2 = (hu)_t + \partial_x (hu^2 + \frac{1}{2}gh^2) + ghS_f - ghS_0$

或者使用速度形式 $u$：
$R_2 = u_t + \partial_x (\frac{1}{2}u^2 + g(h+z_b)) + g n^2 u |u| h^{-4/3}$

我们将采用速度形式，因为它更直接地关联到伯努利方程和变分导数 $\frac{\delta H}{\delta h}$。
