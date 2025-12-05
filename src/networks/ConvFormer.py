import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# ------------------------------------------------------------------
# Sub-components
# ------------------------------------------------------------------

class SpatialEncoderBlock(layers.Layer):
    """Time-Distributed Conv1D ResNet Block"""
    def __init__(self, filters, kernel_size=5, activation='tanh'):
        super().__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same')
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='same')
        self.act = layers.Activation(activation)
        # TimeDistributed wrapper applied in call or parent

    def call(self, inputs):
        # inputs: (Batch * T, Nx, C) - folded time dimension
        res = inputs
        x = self.conv1(inputs)
        x = self.act(x)
        x = self.conv2(x)
        return self.act(x + res)

class TemporalTransformerBlock(layers.Layer):
    """Transformer Encoder Block for Temporal Evolution"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'), 
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # x: (Batch * Nx, T, d_model)
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        # Cast pos_encoding to match inputs dtype (e.g., float16)
        pos_encoding_casted = tf.cast(self.pos_encoding, inputs.dtype)
        return inputs + pos_encoding_casted[:, :tf.shape(inputs)[1], :]

# ------------------------------------------------------------------
# Main Model
# ------------------------------------------------------------------

class ConvFormer(Model):
    def __init__(self, par):
        super(ConvFormer, self).__init__()
        
        arch = par.architecture
        self.d_model = arch.get("d_model", 64)
        self.n_conv = arch.get("n_conv_layers", 3)
        self.n_trans = arch.get("n_transformer_layers", 2)
        self.heads = arch.get("n_heads", 4)
        self.window = arch.get("window_size", 120)
        self.kernel = arch.get("kernel_size", 5)
        
        # 1. Spatial Encoder
        self.input_proj = layers.Dense(self.d_model) # Project to latent dim
        self.spatial_blocks = [
            SpatialEncoderBlock(self.d_model, self.kernel) 
            for _ in range(self.n_conv)
        ]
        
        # 2. Temporal Transformer
        self.pos_encoding = PositionalEncoding(self.window, self.d_model)
        self.trans_blocks = [
            TemporalTransformerBlock(self.d_model, self.heads, self.d_model*2)
            for _ in range(self.n_trans)
        ]
        
        # 3. Spatial Decoder
        self.decoder_conv = layers.Conv1D(self.d_model, self.kernel, padding='same', activation='tanh')
        self.output_proj = layers.Dense(2) # [h, u]
        
        # Shim for Theta
        self._dummy = tf.Variable(0.0, trainable=False)

    def call(self, inputs):
        # inputs: (Batch, T, Nx, 3)
        B = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        Nx = tf.shape(inputs)[2]
        
        # --- A. Time-Distributed Spatial Encoding ---
        # Fold time into batch: (B*T, Nx, 3)
        x = tf.reshape(inputs, [B*T, Nx, 3])
        
        x = self.input_proj(x)
        for block in self.spatial_blocks:
            x = block(x)
            
        # Unfold: (B, T, Nx, d_model)
        x = tf.reshape(x, [B, T, Nx, self.d_model])
        
        # --- B. Temporal Evolution ---
        # Permute to treat each spatial point as a sequence: (B, Nx, T, d_model)
        x = tf.transpose(x, [0, 2, 1, 3])
        # Fold spatial into batch: (B*Nx, T, d_model)
        x = tf.reshape(x, [B*Nx, T, self.d_model])
        
        x = self.pos_encoding(x)
        for block in self.trans_blocks:
            x = block(x)
            
        # Aggregate: Take the LAST time step as the context for next prediction
        # x_last: (B*Nx, d_model)
        x_last = x[:, -1, :]
        
        # Unfold spatial: (B, Nx, d_model)
        x_last = tf.reshape(x_last, [B, Nx, self.d_model])
        
        # --- C. Spatial Decoding ---
        x = self.decoder_conv(x_last)
        delta_U = self.output_proj(x)
        
        # Residual Connection from the LAST input frame
        # Last input frame: inputs[:, -1, :, 0:2]
        U_last = inputs[:, -1, :, 0:2]
        
        return U_last + delta_U

    @property
    def nn_params(self):
        # Compatibility shim
        class ThetaShim:
            def __init__(self, v): self.values = v
        return ThetaShim(self.trainable_variables)

    @nn_params.setter
    def nn_params(self, theta):
        vals = theta.values if hasattr(theta, 'values') else theta
        for var, val in zip(self.trainable_variables, vals):
            var.assign(val)
