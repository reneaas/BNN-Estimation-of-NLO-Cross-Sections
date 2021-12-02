import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class BNNregression(object):
    """Implements a simple Bayesian neural network
    using HMC for sampling the posterier
    """

    def __init__(self, layers: list, lr: float, activation: str):
        super(BNNregression, self).__init__()
        self.layers = [
            DenseLayer(n_rows=n, n_cols=m, activation=activation)
            for n, m in zip(layers[1:-1], layers[:-2])
        ]
        self.layers.append(
            DenseLayer(n_rows=layers[-1], n_cols=layers[-2], activation=None)
        )

        self.n_layers = len(self.layers)
        self.lr = lr

    def __call__(self, x):
        self.x = x
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        return self(x)

    def backward(self, y):
        self.layers[-1].err = self.layers[-1].a - y
        self.layers[-1].db = np.sum(self.layers[-1].err, axis=0)

        self.layers[-1].dw = np.sum(
            np.einsum(
                "...j,...k->...jk",
                self.layers[-1].err,
                self.layers[-2].a,
                optimize=True,
            ),
            axis=0,
        )

        for l in range(self.n_layers - 2, 0, -1):
            self.layers[l].err = (
                np.einsum(
                    "...k,kj->...j",
                    self.layers[l + 1].err,
                    self.layers[l + 1].w,
                    optimize=True,
                )
                * self.layers[l].activation(self.layers[l].z)
            )
            self.layers[l].db = np.mean(self.layers[l].err, axis=0)

            self.layers[l].dw = np.sum(
                np.einsum(
                    "...j,...k->...jk",
                    self.layers[l].err,
                    self.layers[l - 1].a,
                    optimize=True,
                ),
                axis=0,
            )

        self.layers[0].err = np.einsum(
            "...k,kj->...j", self.layers[1].err, self.layers[1].w, optimize=True
        ) * self.layers[0].activation_derivative(self.layers[0].z)
        self.layers[0].db = np.mean(self.layers[0].err, axis=0)

        self.layers[0].dw = np.sum(
            np.einsum("...j,...k->...jk", self.layers[0].err, self.x, optimize=True),
            axis=0,
        )

    def update_layers(self):
        for layer in self.layers:
            layer.w -= self.lr * layer.dw
            layer.b -= self.lr * layer.db

    def fit(self, x, y, epochs=10000):
        for i in trange(epochs):
            self(x)
            self.backward(y)
            self.update_layers()
    



class DenseLayer(object):
    """Implements a densely connected layer for a neural network for regression tasks."""

    def __init__(self, n_cols, n_rows, activation):
        super(DenseLayer, self).__init__()
        self.n_cols = n_cols
        self.n_rows = n_rows

        possible_activations = ["sigmoid", "relu", None]
        # Assign activation function
        if not activation in possible_activations:
            raise ValueError(f"{activation=} is not a supported activation function.")

        if activation == "sigmoid":
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == "relu":
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        else:
            self.activation = lambda x: x  # identity function

        # Initialize parameters of the layer
        self.w = np.random.normal(size=(n_rows, n_cols)) / np.sqrt(n_cols)
        self.b = np.random.normal(size=n_rows)
        self.z = np.zeros_like(self.b)
        self.a = np.zeros_like(self.z)

        # Initialize variables to use with backpropagation
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.err = np.zeros_like(self.a)

    def __call__(self, x):
        self.z = np.einsum("ij,...j->...i", self.w, x, optimize=True) + self.b
        self.a = self.activation(self.z)
        return self.a

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        y = self.sigmoid(x)
        return y * (1 - y)

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def relu_derivative(x):
        return 1.0 * (x > 0)


if __name__ == "__main__":
    f = lambda x: np.sin(x)
    layers = [1, 100, 1]
    bnn = BNNregression(layers, lr=0.0001, activation="relu")
    n_train = 100
    x_train = np.random.normal(0, 2, size=(n_train, 1))
    y_train = f(x_train)
    bnn.fit(x_train, y_train, epochs=10000)

    n_test = 10000
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
    x_test = np.random.normal(0, 2, size=(n_test, 1))
    y_test = f(x_test)
    y_hat = bnn(x_test)
    plt.scatter(x_test, y_hat, label="predictions")
    plt.scatter(x_test, y_test, label="ground truth")
    plt.plot(x, f(x))
    plt.legend()
    plt.show()
