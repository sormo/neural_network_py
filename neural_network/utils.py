import numpy as np
from .config import *
from .utils import *

def get_activation_function(name):
    if name == ACTIVATION_FUNCTION_RELU:
        return relu, relu_backward
    elif name == ACTIVATION_FUNCTION_SIGMOID:
        return sigmoid, sigmoid_backward

    raise ValueError(f'Inavalid name of activation function: {name}')

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    Arguments:
    Z   numpy array of any shape
    Returns:
    A   output of sigmoid(z), same shape as Z
    """
    A = 1/(1+np.exp(-Z))

    return A

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A  Post-activation parameter, of the same shape as Z
    Z  
    """
    A = np.maximum(0,Z)

    return A

def relu_backward(dA, Z):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA   post-activation gradient, of any shape
    Z    
    Returns:
    dZ   Gradient of the cost with respect to Z
    """
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0

    return dZ

def sigmoid_backward(dA, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA  post-activation gradient, of any shape
    Z   
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    return dZ