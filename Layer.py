import numpy as np

class Layer():

    '''
        Layer Object: 

        Constructor parameters: 
        layer_dim - int type, number of neurons in the specific layer
        input_dim - int type, number of neurons from the prev layer, or number of features of X if layer 1
        activation - string type, defaults to relu, other value is sigmoid

        Class attributes:
        self.activation - String type: activation function
        self.layer_dim - Int type: number of neurons in said layer
        self.input_dim - Int type: number of neurons from previous layer, or number of features of X if layer 1
        self.w - nparray type: weight matrix for said layer, shape (layer_dim, input_dim)
        self.b - nparray type: bias vector for said layer, shape(layer_dim, 1)
        
        Class methods:

        self.forward_propagate(input)
            Parameters: input activations, size (n[l-1], m), where n[l-1] is the number of neurons in prev layer
            Return: 
                Activation, Parameters
                
        
    

    
    '''

    def __init__(self, layer_dim, input_dim, activation="relu"):
        self.activation = activation
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        self.w = np.random.randn(self.layer_dim, input_dim) * 0.01 # intializing weights with small random values
        self.b = np.zeros((self.layer_dim, 1))

    

