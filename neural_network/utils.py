import numpy as np
import math
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

def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- output data of shape (output size, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous [(mini_batch_X, mini_batch_Y)]
    """
    m = X.shape[1]                  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
