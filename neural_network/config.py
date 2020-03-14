# Some small number usualy 0.3 or 0.1 or 0.01 or so.
HYPERPARAM_LEARNING_RATE = 'learning_rate'
# Number of learning steps, usually in thousands.
HYPERPARAM_LEARNING_STEPS = 'learning_steps'
#  Optional parameter. Used for L2 regularization. Should be number from 0 to 1
# (0 means no regularization). Usually about 0.7
HYPERPARAM_LAMBDA = 'lambda'
# Optional parameter. Droput keep probability. Number from 0 to 1.
# For example 0.86 means that there is 14% chance of shutting down a neuron.
# Do not use droput during test (inly during training)!
HYPERPARAM_DROPOUT_KEEP_PROB = 'keep_prob'

ACTIVATION_FUNCTION_RELU = 'relu'
ACTIVATION_FUNCTION_SIGMOID = 'sigmoid'
