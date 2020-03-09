import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from utils import plot_decision_boundary
from utils import load_planar_dataset, load_circles_dataset, load_moons_dataset, load_blobs_dataset, load_gaussian_quantiles_dataset, load_no_structure_dataset
from neural_network.neural_network import train_model, forward_pass
from neural_network.config import *

def evaluate_dataset(X, Y):
    # Print the shapes of input data
    shape_X = X.shape
    shape_Y = Y.shape
    m = shape_X[1]  # training set size

    print('The shape of X is: ' + str(X.shape))
    print('The shape of Y is: ' + str(Y.shape))
    print(f'I have m = {m} training examples')

    # Visualize the data:
    Y_flat = Y.ravel()
    plt.scatter(X[0,:], X[1,:], c=Y_flat, s=40, cmap=plt.cm.Spectral)
    plt.show()

    # Train the logistic regression classifier
    clf = sklearn.linear_model.LogisticRegressionCV();
    clf.fit(X.T, Y_flat);

    # Plot the decision boundary for logistic regression
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")

    # Print accuracy
    LR_predictions = clf.predict(X.T)
    print(f'Accuracy of logistic regression: {float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100):.4f}%')

    plt.show()

    ###

    # Build a model with a n_h-dimensional hidden layer
    dimensions = [X.shape[0], 20, 7, 5, Y.shape[0]]
    activations = ['', ACTIVATION_FUNCTION_RELU, ACTIVATION_FUNCTION_RELU, ACTIVATION_FUNCTION_RELU, ACTIVATION_FUNCTION_SIGMOID]
    hyperparams = { HYPERPARAM_LEARNING_RATE: 0.0075, HYPERPARAM_LEARNING_STEPS: 2100 }

    model, costs = train_model(dimensions, activations, hyperparams, X, Y)

    pred = forward_pass(model, X) > 0.5

    # Plot the decision boundary
    plot_decision_boundary(lambda x: forward_pass(model, x.T) > 0.5, X, Y)
    plt.title('Decision Boundary for NN')

    # Print accuracy
    predictions = forward_pass(model, X)
    print(f'Accuracy: {float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100):.4f}%')

    plt.show()

datasets = {
    "flower": load_planar_dataset(),
    "noisy_circles": load_circles_dataset(),
    "noisy_moons": load_moons_dataset(),
    "blobs": load_blobs_dataset(),
    "gaussian_quantiles": load_gaussian_quantiles_dataset(),
    "no_structure": load_no_structure_dataset()
}

evaluate_dataset(*datasets['flower'])
