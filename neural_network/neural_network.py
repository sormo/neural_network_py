import numpy as np
from .config import *
from .utils import get_activation_function

def create_context(dimensions, activations, hyperparams):
    """
    Prepare context for nn.
    Input:
    dimensions   List of numbers representing number of units from input layer,
                 hidden layers to output layer. Form [xn, ..., yn]
    activations  List of strings representing names of activation functions.
                 Form ['', 'relu', 'sigmoid', ...]
    hyperparams  Hyperparameters for nn as dict. Expects key 'rand_init_factor'
    Return:
    List of dictionaries in form
      [{}, {'W':<array>, 'b':<array>, 'activation':(<function>, <function>)}, ...]
    """
    result = [dict()]
    for i in range(1, len(dimensions)):
        W = np.random.randn(dimensions[i], dimensions[i-1]) / np.sqrt(dimensions[i-1])
        b = np.zeros((dimensions[i], 1))
        act = get_activation_function(activations[i])
        result.append({'W': W, 'b': b, 'activation': act})

    return result

def compute_forward(context, layer):
    W = context[layer]['W']
    A_prev = context[layer-1]['A']
    b = context[layer]['b']

    context[layer]['Z'] = np.dot(W, A_prev) + b
    context[layer]['A'] = context[layer]['activation'][0](context[layer]['Z'])

def forward_pass(context, X):
    """
    Perform forward pass on nn. Computes Z and A for layers
    Input:
    context      
    X            np array of input features. Dimensions are (xn, m)
    Return:
    Y
    """
    context[0]['A'] = X

    for i in range(1, len(context)):
        compute_forward(context, i)

    return context[-1]['A']

def compute_cost(context, Y):
    """
    Compute current cost of nn
    Input:
    context  
    Y        np array of outputs in form (yn, m)
    Return:
    cost
    """
    m = Y.shape[1]
    AL = context[-1]['A']

    cost = -(1/m)*np.sum(Y*np.log(AL) + (1 - Y)*np.log(1 - AL))
    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    return cost

def compute_backward(context, layer):
    """
    Compute backward gradients. Expects dA to be already computed.
    """
    # this should be same for all activation matrices
    m = context[layer-1]['A'].shape[1]
    A_prev = context[layer-1]['A']
    W = context[layer]['W']
    dZ = context[layer]['activation'][1](context[layer]['dA'], context[layer]['Z'])

    context[layer]['dZ'] = dZ
    context[layer]['dW'] = (1/m) * np.dot(dZ, A_prev.T)
    context[layer]['db'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    context[layer-1]['dA'] = np.dot(W.T, dZ)

def backward_pass(context, Y):
    """
    Perform backward pass on nn
    Input:
    Y       output vector in form (yn, m)
    """
    # derivative of cost with respect to activation in last layer
    L = len(context) - 1
    AL = context[L]['A']
    context[L]['dA'] = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    for i in reversed(range(1, L + 1)):
        compute_backward(context, i)

def update_parameters(context, hyperparams):
    """
    Update parameters according to computed gradients.
    """
    for i in range(1, len(context)):
        context[i]['W'] = context[i]['W'] - hyperparams[HYPERPARAM_LEARNING_RATE] * context[i]['dW']
        context[i]['b'] = context[i]['b'] - hyperparams[HYPERPARAM_LEARNING_RATE] * context[i]['db']

def train_model(dimensions, activations, hyperparams, X, Y):
    """
    Input:
    dimensions   List of numbers representing number of units from input layer,
                 hidden layers to output layer. Form [xn, ..., yn]
    activations  List of strings representing names of activation functions.
                 Form ['', 'relu', 'sigmoid', ...]
    hyperparams  Hyperparameters for nn as dict. Expects key 'rand_init_factor'
    Return:

    """
    context = create_context(dimensions, activations, hyperparams)
    costs = []

    for i in range(hyperparams[HYPERPARAM_LEARNING_STEPS]):
        forward_pass(context, X)

        if i % 100 == 0:
            costs.append(compute_cost(context, Y))
            print(f'Iteration {i:04d} cost {costs[-1]:.4f}')

        backward_pass(context, Y)

        update_parameters(context, hyperparams)

    return context, costs
