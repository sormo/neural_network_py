# Some small number usualy 0.3 or 0.1 or 0.01 or so.
HYPERPARAM_LEARNING_RATE = 'learning_rate'
# Number of learning steps, usually in thousands.
HYPERPARAM_LEARNING_STEPS = 'learning_steps'
#  Optional parameter. Used for L2 regularization. Should be number from 0 to 1
# (0 means no regularization). Usually about 0.7
HYPERPARAM_LAMBDA = 'lambda'
# Optional parameter. Droput keep probability. Number from 0 to 1.
# For example 0.86 means that there is 14% chance of shutting down a neuron.
# Do not use droput during test (only during training)!
HYPERPARAM_DROPOUT_KEEP_PROB = 'keep_prob'
# Optional parameter. Size of mini-batch used during training.
# Usually 64, 128, 256, 512. If none, whole training set is processed before
# parameters update.
HYPERPARAM_MINI_BATCH_SIZE = 'minibatch_size'
# Optional parameter. Used for gradient descent update with momentum.
# Usually about 0.9
HYPERPARAM_MOMENTUM_RATE = 'momentum_rate'
# Optional parameter. Used for gradient descent update with adam.
# Usually about 0.999. HYPERPARAM_MOMENTUM_RATE must be defined for adam.
HYPERPARAM_ADAM_RATE = 'adam_rate'
# Used during adam update of parameters in gradient descent.
# If not specified default 10^{-8} is used.
HYPERPARAM_EPSILON = 'epsilon'

ACTIVATION_FUNCTION_RELU = 'relu'
ACTIVATION_FUNCTION_SIGMOID = 'sigmoid'
