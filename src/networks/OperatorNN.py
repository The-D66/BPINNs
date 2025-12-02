import tensorflow as tf
from tensorflow.keras import layers, Model
from networks.Theta import Theta

class FeatureEncoder(layers.Layer): 
    """
    Encoder for BCs and ICs to extract multi-scale latent features.
    Uses 1D-CNN structure.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Layer 1 (Shallow / High Frequency)
        self.conv1 = layers.Conv1D(32, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling1D(2, padding='same') 
        self.proj1 = layers.Dense(latent_dim, activation='tanh') 

        # Layer 2 (Mid)
        self.conv2 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling1D(2, padding='same')
        self.proj2 = layers.Dense(latent_dim, activation='tanh')

        # Layer 3 (Deep / Low Frequency)
        self.conv3 = layers.Conv1D(128, 3, activation='relu', padding='same')
        self.proj3 = layers.Dense(latent_dim, activation='tanh')
        
        self.global_pool = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        # inputs: (Batch, Sequence_Length, Features)
        
        # Level 1
        f1 = self.conv1(inputs) 
        z1 = self.proj1(self.global_pool(f1)) 
        p1 = self.pool1(f1) 
        
        # Level 2
        f2 = self.conv2(p1)
        z2 = self.proj2(self.global_pool(f2))
        p2 = self.pool2(f2)
        
        # Level 3
        f3 = self.conv3(p2)
        z3 = self.proj3(self.global_pool(f3))
        
        return [z1, z2, z3] 

class TrunkNetWithSkip(layers.Layer): 
    """
    Trunk Network that processes coordinates (x, t) and receives 
    injected features from Encoders via Skip Connections.
    """
    def __init__(self, latent_dim=64, num_trunk_layers=3, trunk_neurons=64):
        super().__init__()
        self.input_dense = layers.Dense(trunk_neurons, activation='tanh')
        self.hidden_layers = [layers.Dense(trunk_neurons, activation='tanh') for _ in range(num_trunk_layers)]
        self.output_dense = layers.Dense(latent_dim, activation='tanh')
        
    def call(self, coords, fused_encoder_features_list):
        # coords: (Batch * Points_per_case, 2)
        # fused_encoder_features_list: List of [Batch, Dim] tensors
        
        # Expand features to match coords dimension (Batch -> Batch * Points)
        # We assume coords are organized as [Case1_Pts, Case2_Pts, ...]
        # So we repeat each case's feature 'num_points_per_case' times.
        
        total_points = tf.shape(coords)[0]
        num_cases = tf.shape(fused_encoder_features_list[0])[0]
        num_points_per_case = total_points // num_cases
        
        expanded_features = []
        for feat in fused_encoder_features_list:
            # (Batch, Dim) -> (Batch, Points, Dim) -> (Batch*Points, Dim)
            feat_expanded = tf.repeat(feat, repeats=num_points_per_case, axis=0)
            expanded_features.append(feat_expanded)
            
        # Input Layer
        x = self.input_dense(coords) 
        
        # Inject Level 1 features
        if len(expanded_features) > 0:
            x = tf.concat([x, expanded_features[0]], axis=-1)

        # Hidden Layers with Injection
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
            # Inject subsequent levels
            if (i + 1) < len(expanded_features): 
                x = tf.concat([x, expanded_features[i + 1]], axis=-1)
                    
        return self.output_dense(x)

class SaintVenantOperator(Model):
    """
    Full Operator Network:
    BC Encoder + IC Encoder -> Fusion -> TrunkNet -> Decoder -> [h, u]
    """
    def __init__(self, par):
        super().__init__()
        
        # Architecture Hyperparameters from config
        self.latent_dim = par.architecture.get("latent_dim", 64)
        self.n_neurons = par.architecture["n_neurons"]
        # Number of feature levels (default 3 for shallow/mid/deep)
        self.num_levels = 3 
        
        self.bc_encoder = FeatureEncoder(latent_dim=self.latent_dim)
        self.ic_encoder = FeatureEncoder(latent_dim=self.latent_dim)
        
        self.trunk = TrunkNetWithSkip(
            latent_dim=self.latent_dim, 
            num_trunk_layers=self.num_levels, 
            trunk_neurons=self.n_neurons
        ) 
        
        # Decoder: maps latent -> physical [h, u]
        # Output dimension is 2 (h, u)
        self.decoder = layers.Dense(2) 

    def call(self, inputs):
        # inputs is expected to be a list/tuple: [bc_seq, ic_seq, xt_query]
        # BUT: Standard Keras/TF model call usually takes one tensor or a list of tensors.
        # Our Dataset pipeline will likely yield a dictionary or a list.
        
        bc_seq, ic_seq, xt_query = inputs
        
        # 1. Encode
        z_bc_multi = self.bc_encoder(bc_seq) # List of 3 tensors
        z_ic_multi = self.ic_encoder(ic_seq) # List of 3 tensors
        
        # 2. Fuse (Concatenation)
        fused_features = [tf.concat([z_bc, z_ic], axis=-1) 
                          for z_bc, z_ic in zip(z_bc_multi, z_ic_multi)]
        
        # 3. Trunk + Skip Injection
        trunk_out = self.trunk(xt_query, fused_features)
            
        # 4. Decode
        return self.decoder(trunk_out)

    # Add nn_params property to be compatible with BayesNN/Theta structure
    @property
    def nn_params(self):
        weights = [layer.get_weights()[0] for layer in self.layers if len(layer.get_weights())>0]
        biases  = [layer.get_weights()[1] for layer in self.layers if len(layer.get_weights())>0]
        # Note: This simple property might miss nested layer weights (like in FeatureEncoder).
        # A better approach for complex models is to iterate trainable_variables.
        
        flat_params = []
        for v in self.trainable_variables:
            flat_params.append(v)
        return Theta(flat_params)

    @nn_params.setter
    def nn_params(self, theta):
        # This sets weights from a Theta object.
        # We need to ensure theta.values order matches self.trainable_variables
        for var, new_val in zip(self.trainable_variables, theta.values):
            var.assign(new_val)
