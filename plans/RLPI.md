
### 一、 核心架构设计

系统由两个耦合的神经网络组成：

1.  **Solver Network (PINN)**: 负责拟合物理场 $(x,t) \rightarrow (h, u)$。
2.  **Policy Network (RL Agent)**: 负责输出黏性系数 $(S) \rightarrow \mu$。

#### 1\. 网络结构图解

```mermaid
graph TD
    subgraph "Solver (PINN Network)"
    A[输入: x, t] --> B(全连接层 MLP)
    B --> C[输出: h, u]
    C --> D{自动微分求导}
    D --> E[导数特征: u_x, h_x...]
    end

    subgraph "Controller (RL/Policy Network)"
    A --> F(全连接层 MLP)
    E -.-> |关键反馈| F
    note[注意: RL网络必须输入<br>导数特征以感知激波] -.-> F
    F --> G[输出: mu]
    G --> H[激活: Softplus * scale]
    end

    subgraph "Physics Engine (Loss Calculation)"
    C & E & H --> I[带黏性的浅水方程残差]
    I --> J[PDE Loss]
    H --> K[L1 正则化 (稀疏惩罚)]
    end

    J --> |梯度反向传播 1| B
    J & K --> |梯度反向传播 2| F
```

#### 2\. 关键设计决策

  * **输入特征（State）**：这是最关键的一点。RL 网络**不能只输入 $(x,t)$**。因为激波的位置是随解变化的，RL 网络必须“看到”当前的梯度才能决定是否加黏性。
      * **建议输入**：$S = [x, t, u, h, |\partial u/\partial x|, |\partial h/\partial x|]$。
  * **动作输出（Action）**：输出人工黏性系数 $\mu(x,t)$。
      * **约束**：$\mu \ge 0$。建议使用 `Softplus` 或 `Sigmoid` 激活函数，并乘以一个缩放因子 $\mu_{max}$（如 0.1）。
  * **奖励/目标函数（Reward/Loss）**：
      * RL 的目标是：**最小化 PDE 残差**（物理守恒） + **最小化 $\mu$ 的使用量**（防止过度平滑）。

-----

### 二、 数学模型与优化目标

#### 1\. 物理方程（带自适应黏性的浅水方程）

一维浅水方程的动量守恒项通常写作（黏性项加在右端）：
$$ (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x = \frac{\partial}{\partial x}\left( \mu(x,t; \theta_{rl}) \frac{\partial (hu)}{\partial x} \right) $$

  * 注：质量方程通常不加黏性，或者加极小的稳态黏性。

#### 2\. 损失函数设计

这是一个对抗与合作并存的过程：

  * **Solver (PINN) Loss**:
    在 $\mu$ 给定的情况下，寻找 $h, u$ 使得方程残差最小。
    $$ L_{Solver}(\theta_{pinn}) = \| \text{Res}(h, u; \mu_{fixed}) \|^2 + \text{BCs} + \text{ICs} $$

  * **Policy (RL) Loss (即负奖励)**:
    寻找 $\mu$，使得在这个 $\mu$ 下计算出的残差最小，同时 $\mu$ 尽可能小。
    $$ L_{RL}(\theta_{rl}) = \| \text{Res}(h, u; \mu) \|^2 + \lambda \|\mu\|_1 $$

      * **$\lambda$ (稀疏惩罚系数)**：这是控制精度的核心超参数。
          * $\lambda$ 过大 $\to \mu \approx 0 \to$ 激波振荡。
          * $\lambda$ 过小 $\to \mu$ 很大 $\to$ 解被抹平（Smearing）。

-----

### 三、 具体实施步骤与参数迭代

建议采用\*\*“交替训练（Alternating Optimization）”\*\*策略：

#### 阶段一：预热（Warm-up）

  * **动作**：固定 $\mu(x,t) = 0.01$ (常数) 或冻结 RL 网络。
  * **目的**：让 PINN 先学出一个大概的波形。如果一开始就让 RL 介入，由于初始残差极大且梯度混乱，RL 会学到错误的策略（例如全场输出最大黏性）。
  * **迭代**：训练 PINN 约 1000\~2000 epochs。

#### 阶段二：联合迭代（Joint Training Loop）

进入主循环，建议 PINN 更新频率高于 RL 更新频率（例如 5:1），因为物理场的调整需要时间适应黏性的变化。

1.  **Phase 1 - 优化 PINN (Solver Update)**：

      * 通过 RL 网络得到 $\mu$（此时 `detach` 掉 RL 网络的梯度，视为常数）。
      * 计算 PDE 残差。
      * 更新 PINN 参数 $\theta_{pinn}$。
      * *重复 5-10 次*。

2.  **Phase 2 - 优化 RL (Policy Update)**：

      * 再次计算 PDE 残差（此时**保留** RL 网络的计算图）。
      * 计算 RL Loss：$Loss = \text{Residual}^2 + \lambda |\mu|$。
      * 通过自动微分，直接计算 $\frac{\partial Loss}{\partial \mu} \cdot \frac{\partial \mu}{\partial \theta_{rl}}$ 并更新 $\theta_{rl}$。
      * *更新 1 次*。


### 四、 避坑指南与实施建议

1.  **关于“平凡解”问题**：
    如果 `lambda_reg` 设置得太小，Agent 可能会发现：“只要我把黏性开到最大，方程就变成了纯扩散方程，解变得非常平滑，残差自然就很小了”。但这会丢失激波特征。

      * **解决**：必须调节 `lambda_reg`，并严格施加初始条件（IC）损失。初始条件的尖锐度会强迫 Agent 即使在 $t=0$ 附近也不能随意加大黏性。

2.  **特征归一化**：
    输入的梯度 $u_x, h_x$ 在激波处数值可能极大（例如 \>100），直接输入神经网络会导致梯度爆炸或饱和。

      * **解决**：在输入 Agent 前，对特征取 `Tanh` 或 `Log`，例如输入 `torch.tanh(u_x)`。

3.  **因果训练 (Causality)**：
    浅水方程是双曲型的，信息沿特征线传播。如果全时空域一起训练，Agent 很难学。

      * **建议**：用 $L(\theta)=\frac{1}{N_{t}} \sum_{i=1}^{N_{t}} w_{i} L_{r}\left(t_{i}, \theta\right)$ 加权loss