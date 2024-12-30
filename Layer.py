import numpy as np

class Layer():

    '''
        Layer Object: 

            Constructor parameters: 
                layer_dim - int type, number of neurons in the specific layer
                input_dim - int type, number of neurons from the prev layer, or number of features of X if layer 1
                activation - string type, defaults to relu, other value is sigmoid

            Class attributes:
                self.activation - string type: activation function, defaults to relu
                self.layer_dim - int type: number of neurons in said layer
                self.input_dim - int type: number of neurons from previous layer, or number of features of X if layer 1
                self.w - nparray type: weight matrix for said layer, shape (layer_dim, input_dim)
                self.b - nparray type: bias vector for said layer, shape(layer_dim, 1)
                self.z_cache - nparray type: cache for z values to be used in backward prop
                self.dw - nparray type: gradients of weight wrt loss for given layer
                self.db - nparray type: gradients of bias wrt loss for given layer
            
            Constructs a layer in a neural network
        
    '''

    def __init__(self, layer_dim, input_dim, activation="relu"):
        self.activation = activation
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        self.w = np.random.randn(self.layer_dim, input_dim) * 0.01 # intializing weights with small random values
        self.b = np.zeros((self.layer_dim, 1))
        self.z_cache = np.zeros((self.layer_dim, 1)) # will be broadcasted based on no. of training examples per pass 
        self.dw = np.zeros((self.layer_dim, self.input_dim))
        self.db = np.zeros((self.layer_dim, 1))

    def forward_propagate(self, input):
        """
            Parameters: 
                dtype np.array((n[l-1], m)): input -> activation from previous layer (or X)
                
            Return: 
                dtype np.array((n[l], m)): al -> activation from given layer
            
            Computes the forward activation (output) of given layer and caches the z value to self.z_cache via broadcasting
        """
        
        # Assertion statements
        assert isinstance(input, np.ndarray), "Input to layer must be a numpy array!"
        assert input.shape[0] == self.input_dim, "Input dimensions do not match!"
        assert self.activation in ["sigmoid", "relu", "tanh"], "Activation not supported!"
        # Forward prop
        self.z_cache = self.z_cache * 0 + np.dot(self.w, input) + self.b
        match self.activation:
            case "sigmoid":
                al = (np.exp(-self.z_cache) + 1)**-1
            case "relu":
                al = np.maximum(0, self.z_cache)
        return al


    def backward_propagate(self, dal):
        """
            Parameters:
                dtype np.array((n[l], m)): dal -> gradient of activation of layer
            
            Return:
                dtype np.array((n[l-1], m)): dal-1 -> gradient of activation of prev layer
            
            Computes the gradients of the weights and bias of given layer, returning the activation for prev layer in backprop
        """
        pass

    

