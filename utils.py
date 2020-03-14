import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy

def plot_cost(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))

    plt.show()

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1

    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def load_circles_dataset():
    X, Y = sklearn.datasets.make_circles(n_samples=200, factor=.5, noise=.3)
    return X.T, Y.reshape(1, Y.shape[0])

def load_moons_dataset():
    X, Y = sklearn.datasets.make_moons(n_samples=200, noise=.2)
    return X.T, Y.reshape(1, Y.shape[0])

def load_blobs_dataset():
    X, Y = sklearn.datasets.make_blobs(n_samples=200, random_state=5, n_features=2, centers=6)
    return X.T, Y.reshape(1, Y.shape[0]) % 2

def load_gaussian_quantiles_dataset():
    X, Y = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=200, n_features=2, n_classes=2, shuffle=True, random_state=None)
    return X.T, Y.reshape(1, Y.shape[0])

def load_no_structure_dataset():
    X, Y = np.random.rand(200, 2), np.random.rand(200, 1) > 0.5
    return X.T, Y.reshape(1, Y.shape[0])

def load_smaller_circles_dataset():
    X, Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    return X.T, Y.reshape(1, Y.shape[0])

def load_goalkeeper_dataset():
    data = scipy.io.loadmat('goalkeeper_dataset/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    # test_X = data['Xval'].T
    # test_Y = data['yval'].T

    return train_X, train_Y
