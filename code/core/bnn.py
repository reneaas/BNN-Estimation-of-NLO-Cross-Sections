import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(100)


class BNNregression(object):
    """Implements a simple Bayesian neural network
    using HMC for sampling the posterier
    """

    def __init__(
        self, layers: list, lr: float, lamb: float = 1e-3, activation: str = "sigmoid"
    ):
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
        self.lamb = lamb

    def __call__(self, x):
        """Computes a standard forward pass of a densily connected
            neural network.

            Args:
                x (np.ndarray)  : Numpy array with input of shape (batch_size, features)

            Returns:
                x (np.ndarray)  : Result of forward pass of the neural network.
        """
        self.x = x
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x: np.ndarray):
        """Alias for __call__. Computes the forward pass of the neural network
        """
        return self(x)

    def backward(self, y: np.ndarray):
        """Computes the backward pass of a the neural network. Computes gradients
            with respect to each parameter of the network.

            Args:
                y (np.ndarray)  :   targets from the training data.
        """
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
        """Updates the parameters of the network given a previous backward pass.
        """
        if self.lamb:
            for layer in self.layers:
                layer.w -= self.lr * layer.dw + self.lamb * layer.w
                layer.b -= self.lr * layer.db + self.lamb * layer.b
        else:
            for layer in self.layers:
                layer.w -= self.lr * layer.dw
                layer.b -= self.lr * layer.db

    def fit(self, x, y, epochs=10000):
        for i in trange(epochs):
            self(x)
            self.backward(y)
            self.update_layers()


    def potential_energy(self, x: np.ndarray, y: np.ndarray):
        """computes the potential energy of the bayesian neural network model.

            Args:
                x (np.ndarray)  :   Input features
                y (np.ndarray)  :   Targets

            Returns:
                Potential energy of the system.
        """
        V = 0
        if self.lamb:
            for layer in self.layers:
                V += np.einsum("ij,ij->", layer.w, layer.w, optimize=True)
                V += np.einsum("i,i->", layer.b, layer.b, optimize=True)
            V *= self.lamb
        return 0.5 * np.sum((y - self(x)) ** 2, axis=0) + V

    def kinetic_energy(self):
        K = 0
        for l in range(len(self.layers)):
            K += 0.5 * np.einsum("ij,ij->", self.p_w[l], self.p_w[l], optimize=True)
            K += 0.5 * np.einsum("i,i->", self.p_b[l], self.p_b[l], optimize=True)
        return K

    def bayesian_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        num_samples: int,
        L: int,
        eps: float,
        num_burn_in: int = 1000,
    ):
        self.L = L
        self.eps = eps
        self.num_samples = num_samples

        for i in trange(num_burn_in):
            self.hmc_step(x, y, burn_in=True)

        for i in trange(self.num_samples):
            self.hmc_step(x, y, burn_in=False)

    def bayesian_predict(self, x):
        predictions = []
        for i in range(self.num_samples):
            for l, layer in enumerate(self.layers):
                layer.w = layer.samples_w[i]
                layer.b = layer.samples_b[i]
            predictions.append(self(x))
        return predictions


    def hmc_step(self, x: np.ndarray, y: np.ndarray, burn_in: bool):
        # Generate momenta
        self.p_w = [np.random.normal(size=layer.w.shape) for layer in self.layers]
        self.p_b = [np.random.normal(size=layer.b.shape) for layer in self.layers]

        # Copy initial weights and biases.
        w_init = [np.copy(layer.w) for layer in self.layers]
        b_init = [np.copy(layer.b) for layer in self.layers]

        K_init = self.kinetic_energy()
        V_init = self.potential_energy(x, y)

        # perform first step of leapfrog.
        self.p_b = [p_b - 0.5 * self.eps * layer.db for p_b, layer in zip(self.p_b, self.layers)]
        self.p_w = [p_w - 0.5 * self.eps * layer.dw for p_w, layer in zip(self.p_w, self.layers)]
        # for l, layer in enumerate(self.layers):
        #     self.p_b[l] -= 0.5 * self.eps * layer.db
        #     self.p_w[l] -= 0.5 * self.eps * layer.dw

        # Do inner steps of leapfrog
        for i in range(self.L - 1):
            for l, layer in enumerate(self.layers):
                # Update generalized coordinates (parameters of network)
                layer.w += self.eps * self.p_w[l]
                layer.b += self.eps * self.p_b[l]

            # Compute new gradients
            self.forward(x)
            self.backward(y)

            # Update momenta
            self.p_w = [p_w - self.eps * layer.dw for p_w, layer in zip(self.p_w, self.layers)]
            self.p_b = [p_b - self.eps * layer.db for p_b, layer in zip(self.p_b, self.layers)]

        # Perform Final step of leapfrog
        for l, layer in enumerate(self.layers):
            layer.w += self.eps * self.p_w[l]
            layer.b += self.eps * self.p_b[l]

        # Compute new gradients
        self.forward(x)
        self.backward(y)

        # Update final momenta.
        self.p_b = [p_b - 0.5 * self.eps * layer.db for p_b, layer in zip(self.p_b, self.layers)]
        self.p_w = [p_w - 0.5 * self.eps * layer.dw for p_w, layer in zip(self.p_w, self.layers)]

        # Compute final energy and differences
        K_final = self.kinetic_energy()
        V_final = self.potential_energy(x, y)
        dK = K_final - K_init
        dV = V_final - V_init

        # Metropolis Hastings part:
        if not burn_in:
            if np.random.uniform() <= min(1, np.exp(-dV) * np.exp(-dK)):
                # Accept new weights
                for l, layer in enumerate(self.layers):
                    layer.samples_w.append(layer.w)
                    layer.samples_b.append(layer.b)
            else:
                # Keep the old weights
                for l, layer in enumerate(self.layers):
                    layer.samples_w.append(w_init[l])
                    layer.samples_b.append(b_init[l])


class DenseLayer(object):
    """Implements a densely connected layer for a neural network for regression tasks."""

    def __init__(self, n_cols, n_rows, activation):
        super(DenseLayer, self).__init__()
        self.n_cols = n_cols
        self.n_rows = n_rows

        self.samples_w = []
        self.samples_b = []

        possible_activations = ["sigmoid", "relu", "tanh", None]
        # Assign activation function
        if not activation in possible_activations:
            raise ValueError(f"{activation=} is not a supported activation function.")

        if activation == "sigmoid":
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == "relu":
            self.activation = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation == "tanh":
            self.activation = self.tanh
            self.activation_derivative = self.tanh_derivative
        else:
            self.activation = lambda x: x  # identity function
            self.activation_derivative = lambda x: 1

        # Initialize parameters of the layer
        self.w = np.random.normal(size=(n_rows, n_cols)) / np.sqrt(n_cols)
        self.b = np.random.normal(size=n_rows)
        self.z = np.zeros_like(self.b)
        self.a = np.zeros_like(self.z)

        # Initialize variables to use with backpropagation
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.err = np.zeros_like(self.a)

        # Store first set of weights used for the BNN
        self.samples_w.append(self.w)
        self.samples_b.append(self.b)

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

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        y = np.tanh(x)
        return 1 - y ** 2


if __name__ == "__main__":
    f = lambda x: np.sin(x)
    layers = [1, 50, 1]
    bnn = BNNregression(layers, lr=0.0001, lamb=1e-5, activation="sigmoid")
    n_train = 100
    x_train = np.random.normal(0, 2, size=(n_train, 1))
    y_train = f(x_train)
    bnn.fit(x_train, y_train, epochs=1000)
    bnn.bayesian_fit(x=x_train, y=y_train, num_samples=1000, L=60, eps=0.001)

    n_test = 1000
    x_test = np.random.normal(0, 2, size=(n_test, 1))
    y_test = f(x_test)

    predictions = bnn.bayesian_predict(x_test)
    for i in range(len(predictions)):
        predictions[i] = predictions[i].squeeze(axis=-1)

    X = np.array(list(x_test) * bnn.num_samples).squeeze(-1)
    predictions = np.array(predictions)
    predictions = predictions.reshape(n_test * bnn.num_samples, 1).squeeze(axis=-1)
    sns.lineplot(x=X, y=predictions, ci="sd")
    # sns.scatterplot(X, predictions, label="predictions")

    plt.scatter(x_train, y_train, label="observations")
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
    plt.plot(x, f(x), label="True function", color="r")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
