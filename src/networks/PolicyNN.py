import tensorflow as tf
from networks.CoreNN import CoreNN
from networks.Theta import Theta

class PolicyNN(CoreNN):
    """
    Policy Network for RLPI.
    Outputs viscosity coefficient mu given state.
    """
    def __init__(self, par):
        # We initialize with parent, but we will immediately overwrite 
        # the model and parameters because the dimensions are different.
        super().__init__(par)

        # RLPI Specific Parameters
        # Check if they exist in par.utils (from args), else default
        self.mu_max = par.utils.get("mu_max", 0.1) 
        
        # Override Dimensions
        # State: [x, t, u, h, |ux|, |hx|] -> 6 inputs
        self.n_inputs = 6 
        self.n_out_sol = 1
        self.n_out_par = 0
        
        # Rebuild Model
        self.model = self.__build_policy_NN(par.utils["random_seed"])
        self.dim_theta = self.nn_params.size()

    def __build_policy_NN(self, seed):
        """
        Initializes the Policy Network
        """
        tf.random.set_seed(seed + 1000) # Different seed from Solver
        
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.n_inputs,)))
        
        # Hidden Layers (Reuse architecture params from Solver for now)
        for _ in range(self.n_layers):
            initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=self.stddev)
            model.add(tf.keras.layers.Dense(self.n_neurons, activation=self.activation, 
                      kernel_initializer=initializer, bias_initializer='zeros'))
        
        # Output Layer
        # Output is scalar mu.
        # Use Softplus to ensure positivity.
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=self.stddev)
        model.add(tf.keras.layers.Dense(1, 
                      kernel_initializer=initializer, bias_initializer='zeros',
                      activation='softplus'))

        return model

    def forward(self, inputs):
        """ 
        Prediction of mu
        inputs : tf tensor (n_samples, 6)
        output : tf tensor (n_samples, 1) scaled by mu_max
        """
        # inputs should be (N, 6)
        raw_out = self.model(inputs)
        return raw_out * self.mu_max
