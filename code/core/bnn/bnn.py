import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import sys
from typing import Callable, Optional, Union
import itertools
import pandas as pd
import re


tfd = tfp.distributions

np.random.seed(1)
tf.random.set_seed(1)


class BayesianNeuralNetwork:
    """Class for a Bayesian neural network (BNN) using Hamiltonian Monte Carlo (HMC)
    and its derivatives as a sampling method to sample from its posterier.

        Args:
            layers (list, optional)                 :   List containing the nodes of each layer. Shape is
                                                        [input_size, nodes_of_layer1, ..., nodes_of_layerN, num_outputs]
            activation (list[function], optional)   :   List of activation functions, one per each layer.
                                                        len(activation) = len(layers) - 1.
                                                        If set to None, each layer is set to `tf.nn.sigmoid`,
                                                        with the top layer set to `tf.identity`.
            kernel_prior (tfd.Distribution)        :   If set to None, it defaults to tfp.distributions.Normal
            bias_prior (tfd.Distribution)          :   If set to None, it defaults to tfp.distributions.Normal
            prior_mean (float)                      :   Mean value of the tfp.distribution.Normal.
                                                        Default: `prior_mean = 0.0`
            prior_stddev (float)                    :   Standard deviation of tfp.distribution.Normal
                                                        Default: `prior_stddev = 0.01`
            lamb (float)                            :   Regularization parameter. Default `lamb = 0.0`.
            num_chains (int)                        :   Number of chains of the parameters to be sampled using MCMC methods.
                                                        If `num_chains` > 1, the MCMC sampling chain will run multiple chains
                                                        in parallel. Default: `num_chains=1`.
    """

    def __init__(
        self,
        layers=None,
        activation=None,
        kernel_prior=None,
        bias_prior=None,
        lamb=0.0,
        num_chains=1,
    ):
        self.num_chains = num_chains
        self.lamb = lamb

        # Set priors of kernel
        if kernel_prior:
            self.kernel_prior = kernel_prior
        else:
            self.kernel_prior = tfd.Normal(loc=0.0, scale=0.01)

        # Set priors of bias
        if bias_prior:
            self.bias_prior = bias_prior
        else:
            self.bias_prior = tfd.Normal(loc=0.0, scale=0.01)

        # Get initial parameters, if layers are provided.
        if layers is not None:
            self.weights = self._create_layers(layers)

        # Set activation and check for activations
        if activation is not None and isinstance(activation, list) is False:
            try:
                self.activation = [
                    lambda x: activation(x) for _ in range(len(layers) - 2)
                ]
                self.activation.append(tf.identity)
            except ValueError:
                print(f"activation {activation} was not a valid function.")
        elif isinstance(activation, list) is True:
            if len(activation) == len(layers) - 1:
                if activation[-1] is tf.identity:
                    self.activation = [a for a in activation]
                else:
                    self.activation = [a for a in activation[:-1]]
                    self.activation.append(tf.identity)
            else:
                raise ValueError(
                    f"len(activation) ({len(activation)}) != len(layers) - 1 ({len(layers)-1})"
                )
        else:
            self.activation = [tf.nn.sigmoid for _ in range(len(layers) - 1)]
            self.activation.append(tf.identity)

    def _create_layers(self, layers):
        weights = []
        for n, m in zip(layers[:-1], layers[1:]):
            weights.extend(
                [
                    self.kernel_prior.sample(sample_shape=(self.num_chains, n, m)),
                    self.bias_prior.sample(sample_shape=(self.num_chains, m)),
                ]
            )
        return weights

    @tf.function
    def __call__(self, x, weights=None):
        """Performs a simple forward pass with set of weights and a given input x

        Args:
            x (tf.Tensor)                           :   Input features of shape (num_points, num_features)
            weights (optional, list[tf.Tensor])     :   List of weights of the network of shape
                                                        (batch_size, n, m).
                                                        Default: `weights=None`. In this case, it defaults
                                                        to the weights stored in the class.

        Returns:
            The computed outputs of the forward pass. A tf.Tensor object of shape [num_weights, num_points, num_outputs]

        """
        if weights is None:
            kernel = self.weights[::2]
            bias = self.weights[1::2]
        else:
            kernel = weights[::2]
            bias = weights[1::2]

        for w, b, activation in zip(kernel, bias, self.activation):
            x = self.dense_layer(x, w, b, activation)
        return x

    @tf.function
    def train_step(self, x, y):
        """Computes a training step in conjunction with the MLE fit
        method.

        Args:
            x (tf.Tensor)   :   Input features of shape [num_points, num_features]
            y (tf.Tensor)   :   Targets of shape [num_points, num_outputs]

        Returns:
            loss (tf.Tensor)    :   Loss of shape=().
        """
        with tf.GradientTape() as tape:
            tape.watch(self.weights)
            y_pred = self(x, self.weights)
            loss = tf.reduce_sum(
                (y - y_pred) ** 2, axis=(1, 2)
            ) + self.log_prior_prob_fn(self.weights)
        grad = tape.gradient(loss, self.weights)
        self.optimizer.apply_gradients(zip(grad, self.weights))
        return loss

    def mle_fit(
        self,
        x,
        y,
        epochs,
        lr,
        batch_size=None,
        optimizer=None,
    ):
        """Fits the network using the backpropagation algorithm, i.e maximum likelihood estimation (MLE).

        Args:
            x (tf.Tensor)                                           :   Input features of shape [num_points, num_features]
            y (tf.Tensor)                                           :   Targets of shape [num_points, num_outputs]
            epochs (int)                                            :   Number of epochs to fit the network.
            lr (float)                                              :   Learning rate.
            batch_size (int, optional)                              :   Batch size used during training.
            optimizer(tf.keras.optimizers.Optimizer, optional)      :   Optimizer used during training.

        Returns:
            loss (list[float])  : List containing the loss per epoch.
        """
        if len(x.shape) != 2:
            raise ValueError(
                f"""
                len(x.shape) = {len(x.shape)} != 2. The shape of the input features must be (num_points, num_features)
            """
            )
        if len(y.shape) != 2:
            raise ValueError(
                f"""
                len(y.shape) = {len(y.shape)} != 2. The shape of the input targets must be (num_points, num_targets)
            """
            )

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            if isinstance(optimizer, tf.keras.optimizers.Optimizer) is False:
                err_message = "\n".join(
                    [
                        f"optimizer = {optimizer} is not a valid optimizer.",
                        "Provide an instance of tf.keras.optimizers.Optimizer",
                        "or instances of classes derived from it.",
                    ]
                )
                raise ValueError(err_message)
            else:
                self.optimizer = optimizer

        losses = []
        self.weights = [tf.Variable(w) for w in self.weights]
        if batch_size is not None and x.shape[0] > batch_size:
            ds = self.get_dataset(x=x, y=y, batch_size=batch_size)
            for _ in trange(epochs, desc="Epochs"):
                tot_loss = 0
                for x_, y_ in ds:
                    loss = self.train_step(x_, y_)
                    tot_loss += loss
                losses.append(tot_loss)

        else:
            for _ in trange(epochs, desc="Epochs"):
                loss = self.train_step(x, y)
                losses.append(loss)
        return loss

    def log_prior_prob_fn(self, weights):
        """Log prior probability function of the weights of the network.
        Currently set to be exp(-lamb * w ** 2) according to Radford Neals treatment
        of BNNs. `lamb` is the analogue to a regularization
        hyperparameter in L2-regularization and serves the same purpose for BNNs.

        Args:
            weights (list[tf.Tensor])   :   List containing tensors with kernels and biases.

        Returns:
            tf.Tensor with the computed prior log probability. Returned shape: [num_chains]
        """
        kernel = weights[::2]
        bias = weights[1::2]
        kernel_sum = tf.reduce_sum(
            [tf.reduce_sum(w ** 2, axis=(-1, -2)) for w in kernel], axis=0
        )
        bias_sum = tf.reduce_sum([tf.reduce_sum(b ** 2, axis=-1) for b in bias], axis=0)
        return -0.5 * self.lamb * (kernel_sum + bias_sum)

    def log_likelihood_fn(self, x, y, weights):
        """Computes log likelihood of predicted target y given features `x` and `weights`.

        Args:
            x  (tf.Tensor)              :   Input features of shape (num_points, num_features)
            y   (tf.Tensor)             :   Predicted targets of shape (num_points, num_outputs)
            weights (list[tf.Tensor])   :   list of weights of the network.

        Returns:
            The log likelihood of yhat as the mean value given target y.
            Equivalent to the residual sum of squares (RSS). Returned shape [num_chains]
        """
        yhat = self(x, weights)
        return -0.5 * tf.reduce_sum((y - yhat) ** 2, axis=(-1, -2))

    def get_target_log_prob_fn(self, x, y):
        """Returns the combines log probability function needed to
        use tf.mcmc.sample_chain.

        Args:
            x (tf.Tensor)   :   Input features of shape (num_points, num_features).
                                Usually only used with training data.
            y (tf.Tensor)   :   Predicted targets of shape (num_points, num_outputs).
                                Usually only used with training data.

        Returns:
            target_log_prob_fn (target log probability function used in the MCMC chain).
        """

        def target_log_prob_fn(*weights):
            return self.log_prior_prob_fn(weights) + self.log_likelihood_fn(
                x, y, weights
            )

        return target_log_prob_fn

    def dense_layer(self, x, w, b, activation):
        """Computes the output of a dense layer.

        Args:
            x   (tf.Tensor)     :   Input features of shape (num_points, num_features)
            w   (tf.Tensor)     :   Kernel of layer. Shape: (batch_size, n, m)
            b   (tf.Tensor)     :   Bias of layer. Shape: (batch_size, m, )
            activation (tf.nn)  :   Activation function of type tf.nn.
                                    E.g tf.nn.sigmoid or tf.nn.relu.

        Returns:
            Computes activations of shape (batch_size, num_points, num_outputs)
        """
        return activation(tf.matmul(x, w) + b[..., None, :])

    def predict_from_chain(self, x, chain=None):
        """Computes predictions of a given x from from a chain of samples from the MCMC chain.

        Args:
            x   (tf.Tensor)         :   Input features of shape (num_points, num_features)
            chain (optional, list)  :   List of weights samples from the MCMC chain
                                        of length num_kernels + num_biases.
                                        Each kernel is of shape (batch_size, m, n)
                                        Each bias is of shape (batch_size, m)

        Returns
            Predictions (tf.Tensor) :   Tensor with predictions. Shape: (batch_size, num_points, num_outputs).
        """
        if chain is not None:
            predictions = self(x, chain)
        else:
            try:
                predictions = self(x, self.chain)
            except ValueError:
                print(
                    """No chain is available. Please provide a chain or assign 
                    it to the network as an attribute named `chain`."""
                )
        return predictions

    @tf.function
    def _sample_chain(self, *args, **kwargs):
        """A simple wrapper around tfp.mcmc.sample_chain that speeds up code
        by compiling to a computational graph using tf.function.
        """
        return tfp.mcmc.sample_chain(*args, **kwargs)

    def get_dataset(self, x, y, batch_size=16):
        """Creates a tf.data.Dataset object from training data.

        Args:
            x   (tf.Tensor)             :   Training features of shape (num_train, num_features)
            y   (tf.Tensor)             :   Training targets of shape (num_train, num_outputs)
            batch_size (optional, int)  :   Batch size of dataset. Default: batch_size=16.

        Returns:
            ds (tf.data.Dataset)    :   Dataset split into batches of size `batch_size`.
        """
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.batch(batch_size=batch_size)
        return ds

    def sample_chain(
        self,
        kernel,
        num_results,
        num_burnin_steps,
        num_steps_between_results=0,
        trace_fn=None,
        parallel_iterations=10,
    ):
        """Produces a MCMC chain given a transition kernel.

        Args:
            kernel (tfp.mcmc.TransitionKernel): Transition kernel to produce MCMC samples.
            num_results (int): number of results per chain.
            num_burnin_steps (int): Number of burn-in steps per chain
            num_steps_between_results (int): Number of steps in between results. Thinning. 
            trace_fn (callable): Python callable. Trace function that traces the MCMC chain.
            parallel_iterations (int): Number of parallel iterations run on hardware supporting SIMD instructions.
        """

        if isinstance(kernel, tfp.mcmc.TransitionKernel) is False:
            raise TypeError(
                f"""{kernel=} is not a valid kernel. Please provide an
                instance of tfp.mcmc.TransitionKernel.
                """
            )
        self.chain = self._sample_chain(
            num_results=num_results,
            kernel=kernel,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=num_steps_between_results,
            trace_fn=trace_fn,
            parallel_iterations=parallel_iterations,
            current_state=self.weights,
        )
        return self.chain

    def save_model(self, fname_prefix):
        """Saves a model given a chain of network samples

        Args:
            fname_prefix (str) :   filename's prefix. By default it's saved as
                                    .npz file.

            chain (list[tf.Tensor]) :   List containing sampled network parameters.
                                        Each element is a list containing tf.Tensors
                                        which makes up a densely connected neural network.
        """
        return NotImplemented


def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layer.Dense(units=50, input_shape=(1,), activation="relu"))
    model.add(tf.keras.layer.Dense(units=1, activation=None))
    return model


def plot_predictions(model, chain):
    return NotImplemented


def main():
    layers = [1, 10, 1]
    n_train = 1000
    n_dims = 1
    f = lambda x: x * tf.math.sin(x) * tf.math.cos(x)
    x_train = tf.random.normal(shape=(n_train, n_dims), mean=0.0, stddev=3.0)
    y_train = f(x_train)

    num_chains = 100
    num_results = 10
    tot_num_results = num_results * num_chains
    num_burnin_steps = 10
    num_leapfrog_steps = 60
    step_size = 0.01

    bnn = BayesianNeuralNetwork(
        layers=layers,
        activation=tf.nn.tanh,
        lamb=0.01,
        num_chains=num_chains,
    )

    start = time.perf_counter()
    loss = bnn.mle_fit(
        x=x_train,
        y=y_train,
        epochs=1000,
        lr=0.001,
        batch_size=None,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds on MLE fit.")

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=bnn.get_target_log_prob_fn(x_train, y_train),
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
    )

    # kernel = tfp.mcmc.NoUTurnSampler(
    #     target_log_prob_fn=bnn.get_target_log_prob_fn(x_train, y_train),
    #     step_size=step_size,
    # )

    #Adaptive kernel
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps)
    )

    start = time.perf_counter()
    chain = bnn.sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds on mcmc sample chain.")

    # test the model
    n_test = 1000
    x_test = tf.random.normal(shape=(n_test, n_dims), mean=0.0, stddev=3.0)
    predictions = bnn(x_test, weights=chain)

    x = np.array(list(x_test.numpy().squeeze(-1)) * tot_num_results)
    predictions = np.array(predictions)
    predictions = predictions.squeeze(-1).ravel()

    sns.lineplot(x, predictions, ci="sd")
    x = np.linspace(-3 * np.pi, 3 * np.pi, 1001)
    plt.plot(x, f(x), label="True function", color="r")
    # plt.scatter(x_train, y_train, label="observed data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()
