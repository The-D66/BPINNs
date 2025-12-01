import tensorflow as tf
from networks.Theta import Theta

class CoreNN():

    """
    - Builds initial network
    - Contains layers and weights and biases (theta)
    - Does forward pass on given theta
    - Can re-sample theta given a seed
    
    Neural Network parameters:
    nn_params = Theta object, storing weights and biases
    """

    def __init__(self, par):

        # Domain dimensions
        self.n_inputs  = par.comp_dim.n_input
        self.n_out_sol = par.comp_dim.n_out_sol
        self.n_out_par = par.comp_dim.n_out_par

        # Architecture parameters
        self.n_layers   = par.architecture["n_layers"]
        self.n_neurons  = par.architecture["n_neurons"]
        self.activation = par.architecture["activation"]
        
        # Fourier Feature Embedding
        self.fourier_scale = par.architecture.get("fourier_scale", 0.0)
        self.fourier_dim   = par.architecture.get("fourier_dim", 20)
        self.use_fourier   = self.fourier_scale > 0
        
        # Prior standard deviation (regularization strength)
        # Default He-like initialization scaling if not provided
        if "weight_std" in par.uncertainty:
            self.stddev = par.uncertainty["weight_std"]
        else:
            self.stddev = tf.math.sqrt(50.0/self.n_neurons)
        
        # Build the Neural network architecture
        self.model = self.__build_NN(par.utils["random_seed"])
        self.dim_theta = self.nn_params.size()

    # ... properties unchanged ...
    @property
    def nn_params(self):
        """ Getter for nn_params property """
        weights = [layer.get_weights()[0] for layer in self.model.layers]
        biases  = [layer.get_weights()[1] for layer in self.model.layers]
        theta = list()
        for w, b in zip(weights, biases):
            theta.append(w)
            theta.append(b)
        return Theta(theta)

    @nn_params.setter
    def nn_params(self, theta):
        """ Setter for nn_params property """
        for layer, weight, bias in zip(self.model.layers, theta.weights, theta.biases):
            layer.set_weights((weight,bias))

    def __build_NN(self, seed):
        """
        Initializes a fully connected Neural Network with 
        - Random Normal initialization of weights
        - Zero initialization for biases
        """
        # Set random seed for inizialization
        tf.random.set_seed(seed)
        
        # Determine Input Shape
        if self.use_fourier:
            # Fourier features: [sin(Bx), cos(Bx)] -> 2 * fourier_dim
            input_dim = 2 * self.fourier_dim
            # Initialize B matrix (Gaussian sampling)
            self.B = tf.random.normal(shape=(self.n_inputs, self.fourier_dim), stddev=self.fourier_scale, dtype=tf.float32)
        else:
            input_dim = self.n_inputs
            
        # Input Layer
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(input_dim,)))
        # Hidden Layers
        for _ in range(self.n_layers):
            initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=self.stddev)
            model.add(tf.keras.layers.Dense(self.n_neurons, activation=self.activation, 
                      kernel_initializer=initializer, bias_initializer='zeros'))
        # Output Layer
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=self.stddev)
        model.add(tf.keras.layers.Dense(self.n_out_sol+self.n_out_par, 
                      kernel_initializer=initializer, bias_initializer='zeros'))

        return model

    def initialize_NN(self, seed):
        """ Initialization of the Neural Network with given random seed """
        self.model = self.__build_NN(seed)
        return self

    def forward(self, inputs):
        """ 
        Simple prediction on draft of Solution
        inputs : np array  (n_samples, n_input)
        output : tf tensor (n_samples, n_out_sol+n_out_par)
        """
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        
        if self.use_fourier:
            # Apply Fourier Mapping
            # x: (N, n_inputs), B: (n_inputs, fourier_dim)
            # projection: (N, fourier_dim)
            projection = tf.matmul(x, self.B) * 2.0 * 3.1415926535
            # features: (N, 2 * fourier_dim)
            x = tf.concat([tf.sin(projection), tf.cos(projection)], axis=1)
            
        return self.model(x)
