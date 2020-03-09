import matplotlib.pyplot as plt
import numpy as np
import h5py
from neural_network.neural_network import train_model, forward_pass
from utils import plot_cost
from neural_network.config import *

def load_data():
    train_dataset = h5py.File('cat_dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('cat_dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def show_image(index, train_x, train_y, classes):
    plt.imshow(train_x[index])
    print("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    plt.show()

def prepare_data(train_x_orig, train_y, test_x_orig, test_y, verbose=True):
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    if verbose:
        print("Number of training examples: " + str(m_train))
        print("Number of testing examples: " + str(m_test))
        print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print("train_x_orig shape: " + str(train_x_orig.shape))
        print("train_y shape: " + str(train_y.shape))
        print("test_x_orig shape: " + str(test_x_orig.shape))
        print("test_y shape: " + str(test_y.shape))

    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    if verbose:
        print("train_x's shape: " + str(train_x.shape))
        print("test_x's shape: " + str(test_x.shape))

    return train_x, train_y, test_x, test_y

def get_accuracy(model, X, Y):
    m = X.shape[1]
    # number of layers in the neural network
    n = len(model)
    p = np.zeros((1,m))

    # Forward propagation
    Y_pred = forward_pass(model, X)

    # convert probas to 0/1 predictions
    for i in range(0, Y_pred.shape[1]):
        if Y_pred[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    return np.sum(p == Y)/m

train_x, train_y, test_x, test_y, classes = load_data()
train_x, train_y, test_x, test_y = prepare_data(train_x, train_y, test_x, test_y)

dimensions = [12288, 20, 7, 5, 1]
activations = ['', ACTIVATION_FUNCTION_RELU, ACTIVATION_FUNCTION_RELU, ACTIVATION_FUNCTION_RELU, ACTIVATION_FUNCTION_SIGMOID]
hyperparams = { HYPERPARAM_LEARNING_RATE: 0.0075, HYPERPARAM_LEARNING_STEPS: 2100 }

model, costs = train_model(dimensions, activations, hyperparams, train_x, train_y)

print(f'Accurancy train: {get_accuracy(model, train_x, train_y):.4f}')
print(f'Accurancy test: {get_accuracy(model, test_x, test_y):.4f}')

plot_cost(costs, hyperparams[HYPERPARAM_LEARNING_RATE])
