import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
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


class BayesianNeuralNetwork(object):
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

        self.avail_activations = {
            "sigmoid": tf.nn.sigmoid,
            "relu": tf.nn.relu,
            "leaky_relu": tf.nn.leaky_relu,
            "tanh": tf.nn.tanh,
            "identity": tf.identity,
        }

        if activation is not None and layers is not None:

            # Check if list with an activation per layer.
            if isinstance(activation, list):
                assert len(activation) == len(layers) - 1, ValueError(
                    f"""len(activation) = {len(activation)} does not match the number of layers."""
                )

                self.activation = []
                for act in activation:
                    assert type(act) == str or callable(act) is True, TypeError(
                        f"activation = {act} is not of type `str` or is not callable."
                    )

                    assert (
                        act in self.avail_activations
                        or act in self.avail_activations.values()
                    ), ValueError(
                        f"""activation = {act} is not a valid activation. 
                        Available activations:
                        {list(self.avail_activations.keys())}.
                        Or provide the tf.nn equivalent to these. e.g 
                        {list(self.avail_activations.values())}
                        """
                    )
                    if isinstance(act, str):
                        self.activation.append(self.avail_activations.get(act))
                    else:
                        self.activation.append(act)

            # Check if user provides a single activation for the "hidden layers".
            elif isinstance(activation, str):
                assert activation in self.avail_activations, ValueError(
                    f"""activation = {activation} is not a valid activation. 
                    Available activations:
                    {list(self.avail_activations.keys())}.
                    Or provide the tf.nn equivalent to these. e.g 
                    {list(self.avail_activations.values())}
                    """
                )

                self.activation = [
                    self.avail_activations.get(activation)
                    for _ in range(len(layers) - 2)
                ]
                self.activation.append(tf.identity)  # Activation for output layer.

            elif callable(activation) is True:
                assert activation in self.avail_activations.values(), ValueError(
                    f"""activation = {act} is not a valid activation. 
                    Available activations:
                    {list(self.avail_activations.keys())}.
                    Or provide the tf.nn equivalent to these. e.g 
                    {list(self.avail_activations.values())}
                    """
                )

                self.activation = [activation for _ in range(len(layers) - 2)]
                self.activation.append(tf.identity)  # Activation for output layer.
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

            self.weights = self._create_layers(layers)

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
            loss = tf.reduce_mean(
                (y - y_pred) ** 2, axis=(1, 2)
            ) - self.log_prior_prob_fn(self.weights)
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
            ds = self._get_dataset(x=x, y=y, batch_size=batch_size)
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
        return losses

    @tf.function
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

    @tf.function
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

    def _get_dataset(self, x, y, batch_size=None):
        """Creates a tf.data.Dataset object from training data.

        Args:
            x   (tf.Tensor)             :   Training features of shape (num_train, num_features)
            y   (tf.Tensor)             :   Training targets of shape (num_train, num_outputs)
            batch_size (optional, int)  :   Batch size of dataset. Default: batch_size=16.

        Returns:
            ds (tf.data.Dataset)    :   Dataset split into batches of size `batch_size`.
        """
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if batch_size is not None:
            try:
                ds = ds.batch(batch_size=batch_size)
            except TypeError:
                print(f"""batch_size = {batch_size} is not a valid batch size.""")
        return ds

    def set_kernel(self, kernel):
        self.kernel = kernel

    def sample_chain_batched(
        self,
        x,
        y,
        kernel,
        num_results,
        num_burnin_steps,
        num_steps_between_results=0,
        trace_fn=None,
        parallel_iterations=10,
        batch_size=None,
    ):
        # Validate kernel
        if isinstance(kernel, tfp.mcmc.TransitionKernel) is False:
            raise TypeError(
                f"""kernel = {kernel} is not a valid kernel. Please provide an
                instance of tfp.mcmc.TransitionKernel.
                """
            )

        dataset = self._get_dataset(x=x, y=y, batch_size=batch_size)
        self.chain = []
        first_iteration = True
        with tqdm.tqdm(total=len(dataset), desc="Batch") as progress_bar:
            for x_batch, y_batch in dataset:

                # Create new kernel with respect to the batched data set (x_batch, y_batch)
                kernel_params = kernel._parameters
                kernel_params["target_log_prob_fn"] = self.get_target_log_prob_fn(
                    x_batch, y_batch
                )
                kernel = type(kernel)(
                    **kernel_params
                )  # Call constructor of kernel with new target log probability.

                # Sample chain
                chain = self.sample_chain(
                    kernel=kernel,
                    num_results=num_results,
                    num_burnin_steps=num_burnin_steps,
                    num_steps_between_results=num_steps_between_results,
                    trace_fn=trace_fn,
                    parallel_iterations=parallel_iterations,
                )

                if first_iteration is False:
                    self.chain = [
                        tf.concat([old_states, new_states], axis=0)
                        for old_states, new_states in zip(self.chain, chain)
                    ]
                first_iteration = False
                progress_bar.update(1)

        return self.chain

    def sample_chain(
        self,
        kernel,
        num_results,
        num_burnin_steps,
        num_steps_between_results=0,
        trace_fn=None,
        parallel_iterations=10,
        fname=None,
    ):
        """Produces a MCMC chain given a transition kernel.

        Args:
            kernel (tfp.mcmc.TransitionKernel):     Transition kernel to produce MCMC samples.
            num_results (int):                      number of results per chain.
            num_burnin_steps (int):                 Number of burn-in steps per chain
            num_steps_between_results (int):        Number of steps in between results. Thinning.
            trace_fn (callable):                    Python callable. Trace function that traces the MCMC chain.
            parallel_iterations (int):              Number of parallel iterations run on hardware supporting
                                                    SIMD instructions.
            fname (str):                            Filename to save the sample chain to.

        Returns:
            self.chain (list[tf.Tensor])    :   List of MCMC chain.
        """
        if isinstance(kernel, tfp.mcmc.TransitionKernel) is False:
            raise TypeError(
                f"""kernel = {kernel} is not a valid kernel. Please provide an
                instance of tfp.mcmc.TransitionKernel.
                """
            )
        chain = self._sample_chain(
            num_results=num_results,
            kernel=kernel,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=num_steps_between_results,
            trace_fn=trace_fn,
            parallel_iterations=parallel_iterations,
            current_state=self.weights,
        )

        # Restack the chain and assign them as the weights of the model.
        self.weights = self._restack_chain(chain)

        if fname:
            if isinstance(fname, str):
                self._save_model(fname)
            else:
                raise TypeError(
                    f"""fname = {fname} is not of the wrong type. Provide a filename of type `str`."""
                )

        return self.weights

    def _save_model(self, fname):
        """Saves the weights of the model to a numpy zip file (.npz)
        using numpy.savez.

        Args:
            fname (str):    Filename with ending ".npz".

        """
        if fname.endswith(".npz") is False:
            raise ValueError(
                f"""The filename does not end with .npz. Please choose a filename with .npz ending."""
            )
        kernel = self.weights[::2]
        bias = self.weights[1::2]
        weights = {}
        for i, (w, b, activation) in enumerate(zip(kernel, bias, self.activation)):
            weights[f"kernel:{i}"] = w.numpy()
            weights[f"bias:{i}"] = b.numpy()
            if isinstance(activation, str):
                weights[f"activation:{i}"] = activation
            elif callable(activation):
                for key, val in self.avail_activations.items():
                    if activation is val:
                        weights[f"activation:{i}"] = key
        np.savez(file=fname, **weights)

    def load_model(self, fname):
        data = np.load(fname)
        kernel = data.files[::3]
        bias = data.files[1::3]
        activation = data.files[2::3]

        self.weights = []
        self.activation = []
        for kernel_name, bias_name, activation_name in zip(kernel, bias, activation):
            self.weights.extend(
                [
                    tf.convert_to_tensor(data[kernel_name], name=kernel_name),
                    tf.convert_to_tensor(data[bias_name], name=bias_name),
                ]
            )
            self.activation.append(
                self.avail_activations.get(str(data[activation_name]))
            )

    def _restack_chain(self, chain):
        new_chain = []
        for w, b in zip(chain[::2], chain[1::2]):
            new_chain.extend(
                [
                    tf.reshape(
                        tensor=w,
                        shape=(w.shape[0] * w.shape[1], w.shape[2], w.shape[3]),
                    ),
                    tf.reshape(tensor=b, shape=(b.shape[0] * b.shape[1], b.shape[2])),
                ]
            )
        return new_chain

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


def main():
    layers = [1, 50, 1]
    n_train = 1000
    n_dims = 1
    f = lambda x: x * tf.math.sin(x) * tf.math.cos(x)
    x_train = tf.random.normal(shape=(n_train, n_dims), mean=0.0, stddev=3.0)
    y_train = f(x_train) + tf.random.normal(shape=x_train.shape, mean=0.0, stddev=0.5)

    num_chains = 10
    num_results = 100
    tot_num_results = num_results * num_chains
    num_burnin_steps = 1000
    num_leapfrog_steps = 60
    step_size = 0.01

    activation = ["tanh", "identity"]

    bnn = BayesianNeuralNetwork(
        layers=layers,
        activation=activation,
        lamb=0.01,
        num_chains=num_chains,
    )

    start = time.perf_counter()
    loss = bnn.mle_fit(
        x=x_train,
        y=y_train,
        epochs=10000,
        lr=0.001,
        batch_size=None,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds on MLE fit.")

    # plt.plot(loss)
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.show()

    # kernel = tfp.mcmc.HamiltonianMonteCarlo(
    #     target_log_prob_fn=bnn.get_target_log_prob_fn(x_train, y_train),
    #     num_leapfrog_steps=num_leapfrog_steps,
    #     step_size=step_size,
    # )

    kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=bnn.get_target_log_prob_fn(x_train, y_train),
        step_size=step_size,
    )

    # Adaptive kernel
    # kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    #     inner_kernel=kernel, num_adaptation_steps=int(0.8 * num_burnin_steps)
    # )

    start = time.perf_counter()
    chain = bnn.sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=0,
        fname="models/test_no_u_turn.npz",
    )

    # bnn.load_weights(fname="models/test.npz")

    # chain = bnn.sample_chain_batched(
    #     x=x_train,
    #     y=y_train,
    #     kernel=kernel,
    #     num_results=num_results,
    #     num_burnin_steps=num_burnin_steps,
    #     batch_size=1000
    # )

    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds on mcmc sample chain.")

    # test the model
    n_test = 1000
    x_test = tf.random.normal(shape=(n_test, n_dims), mean=0.0, stddev=3.0)
    x_test = tf.linspace(-3 * np.pi, 3 * np.pi, n_test)
    x_test = x_test[:, None]
    start = time.perf_counter()
    predictions = bnn(x_test, weights=chain)
    end = time.perf_counter()
    timeused = end - start
    print(f"timeused = {timeused} seconds on predictions.")
    print(f"{predictions.shape=}")

    predictions = np.array(predictions)
    print(f"{predictions[0, ...].shape=}")
    for i in range(predictions.shape[0]):
        plt.plot(x_test.numpy().squeeze(-1), predictions[i].squeeze(-1), alpha=0.01)
    mean_predictions = np.mean(predictions, axis=0)
    print(f"{mean_predictions.shape=}")
    plt.plot(
        x_test.numpy().squeeze(-1),
        mean_predictions.squeeze(-1),
        label="mean prediction",
    )
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    stddev_predictions = tfp.stats.stddev(
        x=tf.convert_to_tensor(predictions), sample_axis=0
    )
    plt.plot(x_test.numpy().squeeze(-1), mean_predictions.squeeze(-1), color="r")
    plt.fill_between(
        x_test.numpy().squeeze(-1),
        (mean_predictions - stddev_predictions.numpy()).squeeze(-1),
        (mean_predictions + stddev_predictions.numpy()).squeeze(-1),
        alpha=0.2,
    )
    plt.plot(
        x_test.numpy().squeeze(1),
        f(x_test).numpy().squeeze(-1),
        label="ground truth",
        color="g",
    )
    plt.legend()
    plt.show()

    # x = np.array(list(x_test.numpy().squeeze(-1)) * tot_num_results)
    # predictions = predictions.squeeze(-1).ravel()
    # sns.lineplot(x, predictions, ci="sd")
    # x = np.linspace(-3 * np.pi, 3 * np.pi, 1001)
    # plt.plot(x, f(x), label="True function", color="r")
    # # plt.scatter(x_train, y_train, label="observed data")
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()
