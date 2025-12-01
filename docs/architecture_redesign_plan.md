# 圣维南方程求解架构改造方案：基于 Transformer/CNN 的算子学习 (Operator Learning)

## 1. 背景与动机
用户目前的工程场景中，主要的观测数据来自于渠道的上游和下游断面（时间序列数据），以及 $t=0$ 时刻的初始状态（空间分布数据）。传统的 PINN 方法将 BC/IC 视为固定约束或逐点的数据损失，难以泛化到不同的工况。

为了更好地捕捉边界条件（时序特征）和初始条件（空间特征）对流场的影响，并实现对不同工况的快速适应，建议采用 **Operator Learning (DeepONet / FNO 思想)** 的架构。这种架构特别适合处理**多尺度特征**，例如圣维南方程中的洪峰（低频大尺度）和瞬时波动（高频小尺度）。

核心思想是将 **BCs (时间序列)** 和 **ICs (空间序列)** 作为网络的**输入条件 (Conditioning Inputs)**，而不仅仅是训练目标。

---

## 2. 建议的网络架构 (Hybrid Branch-Trunk Architecture with Multi-scale Skip Connections)

我们将构建一个多分支网络架构，分为 **Branch Nets (编码器)** 和 **Trunk Net (坐标网络)**。并融入 **Skip Connection** 思想，以实现多尺度特征的有效融合。

### 2.1 输入定义
1.  **BC Branch (边界条件分支)**:
    *   **输入**: 上游 $[h(0, t), Q(0, t)]$ 和 下游 $[h(L, t), Q(L, t)]$ 的完整时间序列。
    *   **形状**: $(Batch, T_{steps}, 4)$ (假设上下游各2个变量)。
2.  **IC Branch (初始条件分支)**:
    *   **输入**: $t=0$ 时刻全流场的 $[h(x, 0), Q(x, 0)]$ 空间分布。
    *   **形状**: $(Batch, N_{x}, 2)$。
3.  **Trunk (查询坐标)**:
    *   **输入**: 需要查询解的时空坐标 $(x, t)$。
    *   **形状**: $(Batch 	imes 	ext{Points_per_case}, 2)$。注意这里 `Batch` 是工况数量，`Points_per_case` 是每个工况的查询点数量。

### 2.2 编码器设计 (Encoders)

#### A. 边界条件编码器 (BC Encoder) & 初始条件编码器 (IC Encoder) - 推荐 Transformer 或 1D-CNN

考虑到 BCs 和 ICs 都是序列数据，且包含多尺度信息，Encoder 应该设计为能够提取**多尺度潜在特征 (Multi-scale Latent Features)**。

*   **输入**: 序列数据 (例如 BC 时间序列：$(Batch, T_{steps}, 4)$)。
*   **输出**: 一个**特征向量列表**，其中每个向量代表序列在不同抽象层次（尺度）上的编码。例如，`[Z_shallow, Z_mid, Z_deep]`。这些特征将被用于 Trunk Net 的 Skip Connection。
*   **方案 A (Transformer)**:
    *   使用 Positional Encoding 编码时间步 $t$。
    *   通过 Transformer Encoder Layers 提取特征。
    *   可以从不同 Transformer Layer 的输出中提取特征，并通过 Global Pooling 得到多尺度潜在向量。
*   **方案 B (1D-CNN)**:
    *   多层 Conv1D + Pooling 结构，类似 U-Net 的编码部分。
    *   在每一层 Pooling 后，对特征图进行 Global Pooling 并投影到 Latent Space，从而得到多尺度潜在向量。
*   **关于IC是否需要特殊处理**: **是的**。IC 是空间维度的序列，而 BC 是时间维度的序列。物理上，IC 决定了 $t=0$ 的状态，BC 决定了演化过程的强迫项。两者应当通过独立的编码器分别提取多尺度特征。

### 2.3 融合网络 (Fusion / Decoder) - 引入多尺度 Skip Connection

我们将 $Z_{BC}$, $Z_{IC}$ 的多尺度特征与 Trunk Net 对 $(x, t)$ 的嵌入向量 $Z_{coord}$ 进行融合。这里的关键是借鉴 U-Net 的 Skip Connection 思想，将来自编码器的多尺度特征**逐层注入**到 Trunk Net 的不同层次。

*   **融合方式**:
    1.  **编码器特征融合**: 首先，将 BC 编码器和 IC 编码器在同一尺度下提取的特征进行融合（例如拼接 `tf.concat` 或相加 `tf.add`），得到融合后的多尺度特征列表 `Fused_Z_multi`。每个融合后的特征列表元素形状为 `(Batch, 2 * latent_dim)`。
    2.  **Trunk Net 注入**: Trunk Net 的每一层接收查询坐标 $(x, t)$ 的信息，并在其计算过程中，将 `Fused_Z_multi` 中的相应尺度特征注入。
        *   **特征扩展**: 由于 `Fused_Z_multi` 的每个元素形状是 `(Batch, Latent_Dim)`，而 Trunk Net 的输入 `coords` 形状是 `(Batch * Points_per_case, 2)`。因此，在注入到 Trunk Net 之前，编码器输出的特征需要通过 `tf.repeat` 或 `tf.tile` 扩展到与查询点数量匹配的维度，以便进行拼接。
        *   **注入机制**: 通常是**拼接 (Concatenation)**。Trunk Net 的每层输出与对应的扩展后的 `Fused_Z_multi` 特征拼接，再送入下一层。这种逐层注入有助于 Trunk Net 在不同抽象级别上利用来自 BC/IC 的信息。

---

## 3. 改造实施路径

### 步骤 1: 数据管线改造 (Data Pipeline)
目前的 `Dataset` 类是基于点对点 $(x, t) 	o (h, u)$ 的。新的架构需要基于 **"工况" (Case)** 进行组织。

*   **数据结构**:
    *   一个样本 = (完整的BC序列, 完整的IC序列, 查询点集 ${(x, t)}$, 对应的解 ${h, u}$)。
*   **生成器改造**:
    *   修改 `generate_sv_data.py`，使其能够批量生成不同工况（改变 $Q_{in}(t)$, $h_{out}(t)$, $ICs$）。
    *   保存格式需支持 `(N_cases, T_steps, Features)` 用于 BC/IC 序列输入，以及 `(N_cases * Points_per_case, 2)` 用于查询点。

### 步骤 2: 网络模型实现 (src/networks/)

新建 `OperatorNN.py`：

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class FeatureEncoder(layers.Layer): # 编码器：用于BC和IC，提取多尺度特征
    def __init__(self, latent_dim=64):
        super().__init__()
        # 定义多层CNN，每层后进行全局池化并投影，以获取不同尺度的特征
        self.conv1 = layers.Conv1D(32, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling1D(2, padding='same') # padding='same' 确保输出长度不变
        self.proj1 = layers.Dense(latent_dim, activation='tanh') # 投影到统一的 latent_dim

        self.conv2 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling1D(2, padding='same')
        self.proj2 = layers.Dense(latent_dim, activation='tanh')

        self.conv3 = layers.Conv1D(128, 3, activation='relu', padding='same')
        self.proj3 = layers.Dense(latent_dim, activation='tanh')
        
        self.global_pool = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        # inputs 形状例如: (Batch, Sequence_Length, Features)
        
        # 提取浅层特征
        f1 = self.conv1(inputs) 
        z1 = self.proj1(self.global_pool(f1)) # 编码器输出的第一个尺度特征
        p1 = self.pool1(f1) # 池化后进入下一层
        
        # 提取中层特征
        f2 = self.conv2(p1)
        z2 = self.proj2(self.global_pool(f2)) # 编码器输出的第二个尺度特征
        p2 = self.pool2(f2)
        
        # 提取深层特征
        f3 = self.conv3(p2)
        z3 = self.proj3(self.global_pool(f3)) # 编码器输出的第三个尺度特征
        
        return [z1, z2, z3] # 返回多尺度特征列表

class TrunkNetWithSkip(layers.Layer): # 主干网络：处理坐标，并注入多尺度特征
    def __init__(self, latent_dim=64, num_trunk_layers=3, trunk_neurons=64):
        super().__init__()
        # Trunk网络的层数可以与Encoder输出的特征尺度数量对应
        self.input_dense = layers.Dense(trunk_neurons, activation='tanh')
        # 隐藏层列表，其中每层之后可能注入来自Encoder的特征
        self.hidden_layers = [layers.Dense(trunk_neurons, activation='tanh') for _ in range(num_trunk_layers)]
        self.output_dense = layers.Dense(latent_dim, activation='tanh')
        
    def call(self, coords, fused_encoder_features_list):
        # coords: (Batch * Points_per_case, 2)
        # fused_encoder_features_list: 列表 [feat1, feat2, feat3]，每个 feat 的形状为 (Batch, latent_dim)
        
        # 扩展 fused_encoder_features_list 中的每个特征到与 coords 匹配的维度
        num_cases = tf.shape(fused_encoder_features_list[0])[0]
        num_points_per_case = tf.shape(coords)[0] // num_cases
        
        expanded_fused_features = []
        for feat in fused_encoder_features_list:
            # tf.repeat 会在 Batch 维度上重复，确保每个 Case 的特征都对应其所有查询点
            expanded_fused_features.append(tf.repeat(feat, repeats=num_points_per_case, axis=0))
            
        # Trunk 输入层
        x = self.input_dense(coords) 
        
        # [Skip Connection] 注入第一个尺度特征 (浅层特征)
        # 简化表示直接拼接，实际可能需要投影层以匹配维度或进行更复杂的融合
        x = tf.concat([x, expanded_fused_features[0]], axis=-1)

        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
            # 逐层注入后续尺度特征
            if (i + 1) < len(expanded_fused_features): # Trunk 的第 i+1 层注入第 i+1 个编码器特征
                x = tf.concat([x, expanded_fused_features[i + 1]], axis=-1)
                    
        return self.output_dense(x)

class SaintVenantOperator(Model):
    def __init__(self, latent_dim=64, num_encoder_features=3, trunk_neurons=64):
        super().__init__()
        self.bc_encoder = FeatureEncoder(latent_dim=latent_dim)
        self.ic_encoder = FeatureEncoder(latent_dim=latent_dim)
        # Trunk 层数应与编码器输出的特征尺度数量匹配，以便逐层注入
        self.trunk = TrunkNetWithSkip(latent_dim=latent_dim, 
                                      num_trunk_layers=num_encoder_features, 
                                      trunk_neurons=trunk_neurons) 
        self.decoder = layers.Dense(2) # 输出 h, u

    def call(self, inputs):
        # inputs: [bc_seq, ic_seq, xt_query]
        # bc_seq: (Batch, T_steps, 4)
        # ic_seq: (Batch, Nx, 2)
        # xt_query: (Batch * Points_per_case, 2) - 所有工况的查询点拼接在一起
        
        bc_seq, ic_seq, xt_query = inputs
        
        # Encoders返回多尺度特征列表: [z1, z2, z3]，每个元素的形状为 (Batch, latent_dim)
        z_bc_multi = self.bc_encoder(bc_seq)
        z_ic_multi = self.ic_encoder(ic_seq)
        
        # 将IC和BC的多尺度特征逐元素融合 (例如拼接)
        # fused_encoder_features_list 也是列表，每个元素形状为 (Batch, 2 * latent_dim)
        fused_encoder_features_list = [tf.concat([z_bc, z_ic], axis=-1) 
                                       for z_bc, z_ic in zip(z_bc_multi, z_ic_multi)]
        
        # TrunkNetWithSkip 接收坐标和融合后的多尺度特征
        final_trunk_output = self.trunk(xt_query, fused_encoder_features_list)
            
        return self.decoder(final_trunk_output)
```

## 4. 总结
*   **初始条件 (IC)** 应当被视为一种空间特征输入，与作为时间特征的 **边界条件 (BC)** 并列处理。
*   **CNN/Transformer** 的引入是为了从高维序列中提取低维的物理特征向量（Latent Vector），这比直接将序列展平输入 MLP 要高效得多。
*   **多尺度 Skip Connection** 的融合使得网络可以更有效地捕捉和利用数据中的多尺度信息，提高训练稳定性和模型精度。
*   这种架构属于 **Neural Operator** 范畴，特别适合“已知边界/初始条件，求解全场”的工程问题。

## 5. 训练策略详解 (Training Strategy)

针对用户关于“无噪声预训练与贝叶斯微调”的疑问，经过对代码库 (`src/algorithms/Trainer.py`, `HMC.py`) 的检查，确认现有架构支持以下**三阶段训练策略**。

### 阶段 1：确定性预训练 (Deterministic Pre-training)
*   **目标**: 快速学习物理场的**拓扑结构**（波形、传播速度），找到全局最优解的邻域。
*   **算法**: **ADAM** (确定性优化)。
*   **数据**: **无噪声 (Noiseless)** 的大规模合成数据（包含多种工况）。
*   **配置**: `init: "ADAM"`, `losses` 开启数据监督（`data_u`, `data_b`）。
*   **注意**: 在此阶段**不要**使用贝叶斯方法 (HMC/VI)。强行用贝叶斯方法拟合无噪声数据会导致方差参数趋向于负无穷（数值不稳定），且效率极低。

### 阶段 2：权重继承与贝叶斯初始化 (Transfer & Initialization)
*   **机制**: `Trainer.py` 中的 `pre_train()` 方法会将训练好的 ADAM 权重赋值给 `self.model.nn_params`。`HMC` 算法在初始化时 (`sample_theta`) 会直接读取这些权重作为马尔可夫链的起始点。
*   **无需额外操作**: 代码库已原生支持此逻辑。

### 阶段 3：贝叶斯微调 (Bayesian Fine-tuning)
*   **目标**: 适应真实工程数据的**噪声**和**不确定性**。
*   **算法**: **HMC** 或 **VI**。
*   **数据**: **有噪声 (Noisy)** 的真实工程观测数据（通常只有边界条件）。
*   **配置**: `method: "HMC"`, `losses` 关闭内部点数据监督（`data_u`），仅保留边界 (`data_b`) 和 PDE (`pde`)。
*   **原理**: 网络在预训练学到的“形状”基础上，通过 HMC 探索参数分布。由于真实数据存在噪声，网络会自动学习到合理的方差（Uncertainty），从而避免了人工添加噪声可能引入的偏差。

**结论**: 不需要也不建议在预训练阶段人工添加噪声。直接使用“无噪声 ADAM 预训练 -> 真实数据 HMC 微调”是最优路径。
