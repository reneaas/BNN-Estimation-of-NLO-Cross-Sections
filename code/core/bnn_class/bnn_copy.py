import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import sys
from typing import Callable, Optional, Union

np.random.seed(1)
tf.random.set_seed(1)


class BayesianNeuralNetworkHMC:
    """Class for a Bayesian neural network (BNN) using Hamiltonian Monte Carlo (HMC)
    and its derivatives as a sampling method to sample from its posterier.

        Args:
            layers (list, optional)                 :   List containing the nodes of each layer. Shape is
                                                        [input_size, nodes_of_layer1, ..., nodes_of_layerN, num_outputs]
            activation (list[function], optional)   :   List of activation functions, one per each layer.
                                                        len(activation) = len(layers) - 1.
                                                        If set to None, each layer is set to `tf.nn.sigmoid`,
                                                        with the top layer set to `tf.identity`.
            kernel_prior (tfp.distributions)        :   If set to None, it defaults to tfp.distributions.Normal
            bias_prior (tfp.distributions)          :   If set to None, it defaults to tfp.distributions.Normal
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
        layers: Optional[list[int]] = None,
        activation: Optional[Union[list[Callable], Callable]] = None,
        kernel_prior: tfp.distributions.Distribution = None,
        bias_prior: tfp.distributions.Distribution = None,
        prior_mean: float = 0.0,
        prior_stddev: float = 0.01,
        lamb: float = 0.0,
        num_chains: int = 1,
    ):
        self.num_chains = num_chains
        self.lamb = lamb

        # Set priors of kernel
        if kernel_prior:
            self.kernel_prior = kernel_prior
        else:
            self.kernel_prior = tfp.distributions.Normal(
                loc=prior_mean, scale=prior_stddev
            )

        # Set priors of bias
        if bias_prior:
            self.bias_prior = bias_prior
        else:
            self.bias_prior = tfp.distributions.Normal(
                loc=prior_mean, scale=prior_stddev
            )

        # Get initial parameters, if layers are provided.
        if layers is not None:
            self.weights = self._create_layers(layers)

        # Set activation and check for activations
        if activation is not None and not isinstance(activation, list):
            try:
                self.activation = [
                    lambda x: activation(x) for _ in range(len(layers) - 2)
                ]
                self.activation.append(tf.identity)
            except ValueError:
                print(f"activation {activation} was not a valid function.")
        elif isinstance(activation, list):
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

    def _create_layers(self, layers: list[int]):
        tmp = [
            (
                self.kernel_prior.sample(sample_shape=(self.num_chains, n, m)),
                self.bias_prior.sample(sample_shape=(self.num_chains, m)),
            )
            for n, m in zip(layers[:-1], layers[1:])
        ]

        weights = []
        for w in tmp:
            kernel, bias = w
            weights.append(kernel)
            weights.append(bias)
        del tmp  # Delete reference to tmp. Served its purpose.
        return weights

    @tf.function
    def __call__(self, x: tf.Tensor, weights: list[tf.Tensor] = None) -> tf.Tensor:
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
        if weights is not None:
            kernel = weights[::2]
            bias = weights[1::2]
        else:
            kernel = self.weights[::2]
            bias = self.weights[1::2]

        for w, b, activation in zip(kernel, bias, self.activation):
            x = self.dense_layer(x, w, b, activation)
        return x

    @tf.function
    def loss_grad(
        self, x: tf.Tensor, y: tf.Tensor
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        """Computes the gradient given an input (x,y).

        Args:
            x (tf.Tensor)   :   Input features of shape [num_points, num_features]
            y (tf.Tensor)   :   Targets of shape [num_points, num_targets]

        Returns:
            loss (tf.Tensor)        :   loss of shape [num_chains]
            grad (list[tf.Tensor])  :   List of gradients of the loss with respect to
                                        weights of the network.
        """
        with tf.GradientTape() as tape:
            tape.watch(self.weights)
            yhat = self(x, self.weights)
            loss = tf.reduce_sum((y - yhat) ** 2)
        grad = tape.gradient(loss, self.weights)
        return loss, grad

    def mle_fit(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        epochs: int,
        lr: float,
        batch_size: int = None,
        optimizer: tf.keras.optimizers.Optimizer = None,
    ) -> list[tf.Tensor]:
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
        if not optimizer:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        losses = []
        self.weights = [tf.Variable(w) for w in self.weights]
        if batch_size is not None:
            ds = self.get_dataset(x=x, y=y, batch_size=batch_size)
            for _ in trange(epochs, desc="Epochs"):
                tot_loss = 0
                for x_, y_ in ds:
                    loss, grad = self.loss_grad(x_, y_)
                    optimizer.apply_gradients(zip(grad, self.weights))
                    tot_loss += loss
                losses.append(tot_loss)

        else:
            for _ in trange(epochs, desc="Epochs"):
                loss, grad = self.loss_grad(x, y)
                optimizer.apply_gradients(zip(grad, self.weights))
                losses.append(loss)
        return loss

    @tf.function
    def prior_log_prob_fn(self, weights: list[tf.Tensor]) -> tf.Tensor:
        """Log prior probability function of the weights of the network.
        Currently set to be exp(-lamb * w ** 2) according to Radford Neals treatment
        of BNNs. `lamb` is the analogue to a regularization
        hyperparameter in L2-regularization and serves the same purpose for BNNs.

        Args:
            weights (list[tf.Tensor])   :   List containing tensors with kernels and biases.

        Returns:
            tf.Tensor with the computed prior log probability.
        """
        kernel = weights[::2]
        bias = weights[1::2]
        kernel_sum = tf.reduce_sum(
            [tf.reduce_sum(w ** 2) for w in kernel]
        )
        bias_sum = tf.reduce_sum([tf.reduce_sum(b ** 2) for b in bias])
        return -0.5 * self.lamb * (kernel_sum + bias_sum)

    @tf.function
    def log_likelihood(
        self, x: tf.Tensor, y: tf.Tensor, weights: list[tf.Tensor]
    ) -> tf.Tensor:
        """Computes log likelihood of predicted target y given features `x` and `weights`.

        Args:
            x  (tf.Tensor)              :   Input features of shape (num_points, num_features)
            y   (tf.Tensor)             :   Predicted targets of shape (num_points, num_outputs)
            weights (list[tf.Tensor])   :   list of weights of the network.

        Returns:
            The log likelihood of yhat as the mean value given target y.
            Equivalent to the residual sum of squares (RSS).
        """
        yhat = self(x, weights)
        return -0.5 * tf.reduce_sum((y - yhat) ** 2)

    # @tf.function
    def create_log_prob_fn(self, x: tf.Tensor, y: tf.Tensor) -> Callable:
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
            return self.prior_log_prob_fn(weights) + self.log_likelihood(x, y, weights)
        return target_log_prob_fn

    @tf.function
    def dense_layer(
        self, x: tf.Tensor, w: tf.Tensor, b: tf.Tensor, activation: Callable
    ) -> tf.Tensor:
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

    def build_model(self, weights: list[tf.Tensor]) -> Callable:
        """Creates a feed-forward densely connected neural network
        given input weights.

        Args:
            weights (list)  :   List containing the weights of the network.

        Returns:
            model (function)    :   Python callable that computes a forward pass
                                    of the densely connected neural network.
        """

        @tf.function
        def model(x: tf.Tensor):
            kernel = weights[::2]
            bias = weights[1::2]
            for w, b, activation in zip(kernel, bias, self.activation):
                x = self.dense_layer(x, w, b, activation)
            return x

        return model

    def predict_from_chain(
        self, x: tf.Tensor, chain: list[tf.Tensor] = None
    ) -> tf.Tensor:
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
                print("No chain is available. Please provide a chain or assign it to the network as an attribute named `chain`.")
        return predictions

    @tf.function
    def sample_chain(self, *args, **kwargs):
        """A simple wrapper around tfp.mcmc.sample_chain that speeds up code
        by compiling to a computational graph using tf.function.
        """
        return tfp.mcmc.sample_chain(*args, **kwargs)

    def get_dataset(
        self, x: tf.Tensor, y: tf.Tensor, batch_size: int = 16
    ) -> tf.data.Dataset:
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

    def no_u_turn_chain(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        num_results_per_chain: int,
        num_burnin_steps: int,
        step_size: float,
        adaptation: str = None,
    ) -> list[tf.Tensor]:
        """Runs the MCMC chain using the No U-turn sampler.

        Args:
            x   (tf.Tensor)             :   Training features of shape (num_points, num_features)
            y   (tf.Tensor)             :   Training targets of shape (num_points, num_outputs)
            num_burnin_steps (int)      :   Number of burn-in steps in the MCMC chain.
            step_size   (float)         :   Step size used with the No U-turn sampler.

        Returns:
            self.chain (list)           :   List of sampled network parameters. Each element
                                            in the list is a densely connected neural network.
        """
        current_state = self.weights
        target_log_prob_fn = self.create_log_prob_fn(x, y)
        kernel = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn, step_size=step_size
        )

        if adaptation == "simple":
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=kernel,
                num_adaptation_steps=int(0.8 * num_burnin_steps),
                target_accept_prob=0.65,
            )
        elif adaptation == "dual":
            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=kernel,
                num_adaptation_steps=int(0.8 * num_burnin_steps),
                target_accept_prob=0.65,
            )

        self.chain = self.sample_chain(
            kernel=kernel,
            current_state=current_state,
            num_results=num_results_per_chain,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=0,
            trace_fn=None,
        )
        return self.chain

    def hmc_chain(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        num_results_per_chain: int,
        num_burnin_steps: int,
        num_leapfrog_steps: int,
        step_size: float,
        adaptation: str = None,
    ) -> list[tf.Tensor]:
        """Runs the MCMC chain using Hamiltonian Monte Carlo (HMC).

        Args:
            x   (tf.Tensor)             :   Training features of shape (num_points, num_features)
            y   (tf.Tensor)             :   Training targets of shape (num_points, num_outputs)
            num_burnin_steps (int)      :   Number of burn-in steps in the MCMC chain.
            num_leapfrog_steps (int)    :   Number of leapfrog steps in HMC.
            step_size   (float)         :   Step size used in the leapfrog scheme in HMC.

        Returns:
            self.chain (list[tf.Tensor])    :   List of sampled network parameters.
                                                The elements of the list is structured as follows:
                                                [kernel0, bias0, kernel1, bias1, ..., kernelN, biasN]

        """
        current_state = self.weights
        target_log_prob_fn = self.create_log_prob_fn(x, y)
        print(f"{tf.size(target_log_prob_fn(*current_state))=}")
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            store_parameters_in_results=None,
        )
        if adaptation == "simple":
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=kernel,
                num_adaptation_steps=int(0.8 * num_burnin_steps),
                target_accept_prob=0.65,
            )
        elif adaptation == "dual":
            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=kernel,
                num_adaptation_steps=int(0.8 * num_burnin_steps),
                target_accept_prob=0.65,
            )

        self.chain = self.sample_chain(
            kernel=kernel,
            current_state=current_state,
            num_results=num_results_per_chain,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=0,
            trace_fn=None,
        )
        return self.chain

    def save_model(self, fname_prefix: str, chain: list[tf.Tensor]):
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


def get_posterior_model(chain):
    layers = []
    kernel = chain[::2]
    bias = chain[1::2]

    layers = []
    for w, b in zip(kernel[:-1], bias[:-1]):
        pass
    return None


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        layers = [1, 10, 1]

        # Create training data
        n_train = 1000
        dims = 1
        f = lambda x: tf.math.sin(x) * tf.math.cos(x)
        x_train = tf.random.normal(shape=(n_train, dims), mean=0.0, stddev=3.0)
        y_train = f(x_train)

        # Should not be needed.
        # kernel_prior = tfp.distributions.Normal(loc=0., scale=0.01)
        # bias_prior = tfp.distributions.Normal(loc=0., scale=0.01)
        num_chains = 4
        num_results_per_chain = 500
        num_results = num_results_per_chain * num_chains
        num_burnin_steps = 10000
        num_leapfrog_steps = 60
        step_size = 0.01
        bnn = BayesianNeuralNetworkHMC(
            layers=layers, activation=tf.nn.sigmoid, lamb=1e-3, num_chains=num_chains
        )
        yhat = bnn(x_train)
        print(yhat.shape)

        start = time.perf_counter()
        bnn.mle_fit(x_train, y_train, epochs=500, lr=0.1, batch_size=1000)
        end = time.perf_counter()
        timeused = end - start
        print(f"{timeused=} seconds of mle fit")

        start = time.perf_counter()
        chain = bnn.hmc_chain(
            x=x_train,
            y=y_train,
            num_results_per_chain=num_results_per_chain,
            num_burnin_steps=num_burnin_steps,
            num_leapfrog_steps=num_leapfrog_steps,
            step_size=step_size,
            adaptation=None,
        )

        # chain = bnn.no_u_turn_chain(
        #     x=x_train,
        #     y=y_train,
        #     num_results_per_chain=num_results_per_chain,
        #     num_burnin_steps=num_burnin_steps,
        #     step_size=step_size,
        #     adaptation=None,
        # )

        end = time.perf_counter()
        timeused = end - start
        print(f"{timeused=} seconds of hmc sampling")

        # test the model
        n_test = 1000
        x_test = tf.random.normal(shape=(n_test, 1), mean=0.0, stddev=3.0)
        predictions = bnn.predict_from_chain(x_test)
        print(f"{x_test.shape=}")
        print(f"{predictions.shape=}")

        x = np.array(list(x_test.numpy().squeeze(-1)) * num_results)
        predictions = np.array(predictions)
        predictions = predictions.squeeze(-1).ravel()
        print(f"{x.shape=}")
        print(f"{predictions.shape=}")
        sns.lineplot(x, predictions, ci="sd")

        x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
        plt.plot(x, f(x), label="True function", color="r")
        plt.scatter(x_train, y_train, label="observed data")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # Plot histogram of a kernel
        num_params = 3
        print(chain[0].shape)
        for i in range(num_params):
            param = chain[0][:, :, 0, i].numpy().ravel()
            sns.kdeplot(param, fill=True)
        plt.show()
        # print(param)
