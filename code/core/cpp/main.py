import numpy as np
from py_layer import PyLayer
from py_bnn import PyBNN
import matplotlib.pyplot as plt

import time 


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def get_weights(layers):
    weights = []
    for m, n in zip(layers[1:], layers[:-1]):
        w = np.random.normal(size=(m, n))
        b = np.random.normal(size=m)
        weights.extend([w, b])
    return weights


def main():
    layer = PyLayer(n_rows=5, n_cols=1, activation="linear".encode())
    layers = [1, 5, 1]
    bnn = PyBNN(layers=layers, activation="sigmoid".encode())



    weights = bnn.get_weights()
    weights = [np.asarray(w) for w in weights]
    print([w.shape for w in weights])
    x = np.array([0.1])
    y = bnn.forward(x)

    weights = get_weights(layers=layers)
    kernel = weights[::2]
    bias = weights[1::2]
    start = time.perf_counter()
    for w, b in zip(kernel[:-1], bias[:-1]):
        x = sigmoid(np.matmul(w, x) + b)
    y = np.matmul(kernel[-1], x) + bias[-1]
    end = time.perf_counter()
    timeused = end - start
    print(f"{y=} with python")
    print(f"{timeused=} with python")


    new_weights = [w.ravel() for w in weights]
    bnn.weights = new_weights
    x = [0.1]
    start = time.perf_counter()
    y = bnn.forward(x)
    end = time.perf_counter()
    timeused = end - start
    print(f"{y=} with cpp")
    print(f"{timeused=} with cpp")



    layers = [1, 20, 20, 1]
    bnn = PyBNN(layers=layers, activation="tanh".encode())

    f = lambda x: x * np.sin(x) * np.cos(x)
    x_train = np.random.normal(size=1)
    y_train = f(x_train)

    y_pred = bnn.forward(x_train)
    bnn.backward(x_train, y_train)
    gradients = bnn.gradients

    X_train = np.random.normal(loc=0.0, scale=2.0, size=(1000, 1))
    Y_train = f(X_train)
    start = time.perf_counter()
    bnn.mle_fit(X=X_train, Y=Y_train, num_epochs=1000, lr=0.0001)
    end = time.perf_counter()
    timeused = end - start 
    print(f"{timeused=} seconds")

    # X_test = np.random.normal(loc=0.0, scale=2.0, size=(1000, 1))
    X_test = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
    true_vals = f(X_test)
    X_test = X_test[:, None]
    Y_test = f(X_test)

    predictions = np.zeros(shape=(X_test.shape[0], 1))
    for i, x in enumerate(X_test):
        predictions[i, ...] = bnn.forward(x)
    
    plt.plot(X_test.ravel(), predictions.ravel())
    plt.plot(X_test.ravel(), true_vals, label="True function")
    # plt.plot(X_test.ravel(), Y_test.ravel(), label="True function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


    



    



if __name__ == "__main__":
    main()