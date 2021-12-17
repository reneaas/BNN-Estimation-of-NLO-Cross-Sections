import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import sys

np.random.seed(1)
tf.random.set_seed(1)


class BNNDense(tf.keras.layers.Dense):
    """Wrapper for Sequential model that modifies the `set_weights` method to
    allow for change of parameters in the computational graph.
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(BNNDense, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def set_weights(self, kernel, bias):
        self._weights = [kernel, bias]
        self._kernel = tf.Variable(kernel)
        self._bias = tf.Variable(bias)
        print(type(self._kernel))
        #self._weights = [tf.Variable(kernel), tf.Variable(bias)]


class BayesianNeuralNetworkHMC(tf.keras.Sequential):
    """Class for a Bayesian neural network (BNN) using Hamiltonian Monte Carlo (HMC)
    and its derivatives as a sampling method to sample from its posterier.

        Args:
            layers (list, optional)             :   List containing the nodes of each layer. Shape is
                                                    [input_size, nodes_of_layer1, ..., nodes_of_layerN, num_outputs]
            activation (str, optional)          :   Specifying activation function. Options:
                                                    `sigmoid`, `relu`, `leaky_relu`.
                                                    Default: None.
                                                    The hidden layers are then set to `sigmoid`.
            top_layer_activation                :   Default: tf.identity
            kernel_prior (tfp.distributions)    :   If set to None, it defaults to tfp.distributions.Normal
            bias_prior (tfp.distributions)      :   If set to None, it defaults to tfp.distributions.Normal
            prior_mean (float)                  :   Mean value of the tfp.distribution.Normal.
                                                    Default: `prior_mean = 0.0`
            prior_stddev (float)                :   Standard deviation of tfp.distribution.Normal
                                                    Default: `prior_stddev = 0.01`
            lamb (float)                        :   Regularization parameter. Default `lamb = 0.0`.
            batch_size (int)                    :   Batch size of the parameters to be updated using MCMC methods.
                                                    If `batch_size` > 1, the MCMC chains will run multiple chains
                                                    in parallell. Default: `batch_size=1`.
    """

    def __init__(
        self,
        layers=None,
        input_shape=None,
        kernel_prior=None,
        bias_prior=None,
        prior_mean=0.0,
        prior_stddev=0.01,
        lamb=0.0,
        batch_size=1,
    ):

        super(BayesianNeuralNetworkHMC, self).__init__(layers=layers, name=None)

        self.batch_size = (
            batch_size  # Parameter batch_size. Number of independent chains.
        )
        self.lamb = lamb  # Regularization parameter.
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

    # def create_layers(self, layers, input_shape, activation):
    #     self.add(tf.keras.layers.Dense(units=layers[0], input_shape=input_shape, activation=activation))
    #     for n in layers[1:-1]:
    #         self.add(tf.keras.layers.Dense(units=n, activation=activation))
    #     self.add(tf.keras.layers.Dense(units=layers[-1], activation=None)) #Default for regression

    def create_layers(self, layers, input_shape, activation):
        self.add(
            BNNDense(units=layers[0], input_shape=input_shape, activation=activation)
        )
        for units in layers[1:-1]:
            self.add(BNNDense(units=units, activation=activation))
        self.add(BNNDense(units=layers[-1], activation=None))

    def set_weights(self, weights):
        kernel = weights[::2]
        bias = weights[1::2]
        for layer, w, b in zip(self.layers, kernel, bias):
            layer.set_weights(w, b)

    def _create_layers(self, layers):
        tmp = [
            (
                self.kernel_prior.sample(sample_shape=(self.batch_size, n, m)),
                self.bias_prior.sample(sample_shape=(self.batch_size, m)),
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

    # def __call__(self, x, weights=None):
    #     """Performs a simple forward pass with set of weighs and a given input x

    #     Args:
    #         x (tf.Tensor)                           :   Input features of shape (num_points, num_features)
    #         weights (optional, list[tf.Tensor])     :   List of weights of the network of shape
    #                                                     (batch_size, n, m).
    #                                                     Default: `weights=None`. In this case, it defaults
    #                                                     to the weights stored in the class.

    #     Returns:
    #         The computed outputs of the forward pass. A tf.Tensor object of shape [num_weights, num_points, num_outputs]

    #     """
    #     if weights:
    #         model = self.build_model(weights)
    #     else:
    #         model = self.build_model(self.weights)
    #     return model(x)

    @tf.function
    def loss_grad(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(self.weights)
            yhat = self(x)
            loss = tf.reduce_sum((y - yhat) ** 2, axis=(1, 2))
        grad = tape.gradient(loss, self.weights)
        return loss, grad

    def mle_fit(self, x, y, epochs, lr, batch_size=None):

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        losses = []
        self.weights = [tf.Variable(w) for w in self.weights]
        if batch_size:
            ds = self.get_dataset(x=x, y=y, batch_size=batch_size)
            for i in trange(epochs, desc="Epochs"):
                tot_loss = 0
                for x_, y_ in ds:
                    loss, grad = self.loss_grad(x_, y_)
                    optimizer.apply_gradients(zip(grad, self.weights))
                    tot_loss += loss
                losses.append(tot_loss)

        else:
            for i in trange(epochs, desc="Epochs"):
                loss, grad = self.loss_grad(x, y)
                optimizer.apply_gradients(zip(grad, self.weights))
                losses.append(loss)
        return loss

    #@tf.function
    def prior_log_prob_fn(self, weights):
        """Log prior probability function of the weights of the network.
        Currently set to be exp(-lamb * w ** 2) according to Radford Neals treatment
        of BNNs. `lamb` is the analogue to a regularization
        hyperparameter in L2-regularization and serves the same purpose for BNNs.

        Args:
            weights (list[tf.Tensor])   :   List containing tensors with kernels and biases.
        """
        kernel = weights[::2]
        bias = weights[1::2]
        kernel_sum = tf.reduce_sum(
            [tf.reduce_sum(w ** 2) for w in kernel]
        )
        bias_sum = tf.reduce_sum([tf.reduce_sum(b ** 2) for b in bias])
        return -0.5 * self.lamb * (kernel_sum + bias_sum)

    def log_likelihood(self, x, y, weights):
        """Computes log likelihood of predicted target y given features `x` and `weights`.

        Args:
            x  (tf.Tensor)              :   Input features of shape (num_points, num_features)
            y   (tf.Tensor)             :   Predicted targets of shape (num_points, num_outputs)
            weights (list[tf.Tensor])   :   list of weights of the network.

        Returns:
            The log likelihood of yhat as the mean value given target y.
            Equivalent to the residual sum of squares (RSS).
        """

        # model = self.build_model(weights)
        self.set_weights(weights)
        yhat = self(x)
        return -0.5 * tf.reduce_sum((y - yhat) ** 2, axis=0)

    def create_log_prob_fn(self, x, y):
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

    def build_model(self, weights):
        """Creates a feed-forward densely connected neural network
        given input weights.

        Args:
            weights (list)  :   List containing the weights of the network.

        Returns:
            model (function)    :   Python callable that computes a forward pass
                                    of the densely connected neural network.
        """

        def model(x):
            kernel = weights[::2]
            bias = weights[1::2]
            for w, b in zip(kernel[:-1], bias[:-1]):
                x = self.dense_layer(x, w, b, self.activation)
            x = self.dense_layer(x, kernel[-1], bias[-1], self.top_layer_activation)
            return x

        return model

    def predict_from_chain(self, x, chain=None):
        """Computes predictions of a given x from from a chain of samples from the MCMC chain.

        Args:
            x   (tf.Tensor)         :   Input features of shape (num_points, num_features)
            chain (optional, list)  :   List of weights samples from the MCMC chain
                                        of length num_kernels + num_biases.
                                        Each kernel is of shape (batch_size, m, n)
                                        Each bias is of shape (batch_size, m)

        Returns
            Predictions (list)      :   List of predictions. Shape: (batch_size, num_points, num_outputs).
        """
        if chain:
            predictions = self(x, chain)
        else:
            predictions = self(x, self.chain)
        return predictions

    @tf.function
    def sample_chain(self, *args, **kwargs):
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

    def no_u_turn_chain(
        self, x, y, num_results_per_batch, num_burnin_steps, step_size, adaptation=None
    ):
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
                num_adaptation_steps=10,
            )
        elif adaptation == "dual":
            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=kernel,
                num_adaptation_steps=10,
            )

        self.chain = self.sample_chain(
            kernel=kernel,
            current_state=current_state,
            num_results=num_results_per_batch,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=0,
            trace_fn=None,
        )
        return self.chain

    def hmc_chain(
        self,
        x,
        y,
        num_results_per_batch,
        num_burnin_steps,
        num_leapfrog_steps,
        step_size,
        adaptation=None,
    ):
        """Runs the MCMC chain using Hamiltonian Monte Carlo (HMC).

        Args:
            x   (tf.Tensor)             :   Training features of shape (num_points, num_features)
            y   (tf.Tensor)             :   Training targets of shape (num_points, num_outputs)
            num_burnin_steps (int)      :   Number of burn-in steps in the MCMC chain.
            num_leapfrog_steps (int)    :   Number of leapfrog steps in HMC.
            step_size   (float)         :   Step size used in the leapfrog scheme in HMC.

        Returns:
            self.chain (list)           :   List of sampled network parameters. Each element
                                            in the list is a densely connected neural network.
        """
        current_state = self.weights
        target_log_prob_fn = self.create_log_prob_fn(x, y)
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            store_parameters_in_results=None,
        )
        if adaptation == "simple":
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=kernel,
                num_adaptation_steps=10,
            )
        elif adaptation == "dual":
            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=kernel,
                num_adaptation_steps=10,
            )

        self.chain = self.sample_chain(
            kernel=kernel,
            current_state=current_state,
            num_results=num_results_per_batch,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=0,
            trace_fn=None,
        )
        return self.chain


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        layers = [50, 1]
        input_shape = (1,)

        # Create training data
        n_train = 1000
        dims = 1
        f = lambda x: tf.math.sin(x) * tf.math.cos(x)
        x_train = tf.random.normal(shape=(n_train, dims), mean=0.0, stddev=3.0)
        y_train = f(x_train)

        # Should not be needed.
        # kernel_prior = tfp.distributions.Normal(loc=0., scale=0.01)
        # bias_prior = tfp.distributions.Normal(loc=0., scale=0.01)

        batch_size = 1
        num_results_per_batch = 10
        num_results = num_results_per_batch * batch_size
        num_burnin_steps = 1000
        num_leapfrog_steps = 60
        step_size = 0.001
        bnn = BayesianNeuralNetworkHMC(layers=None, lamb=1e-3, batch_size=batch_size)
        bnn.create_layers(layers, input_shape, activation="sigmoid")
        yhat = bnn(x_train)
        print(yhat.shape)

        weights = bnn.get_weights()
        new_weights = [tf.Variable(tf.random.normal(w.shape)) for w in weights]
        print(new_weights[0].shape)
        #bnn.set_weights(new_weights)

        target_prob_fn = bnn.create_log_prob_fn(x_train, y_train)
        p = target_prob_fn(new_weights)
        print(p)

        #yhat = bnn(x_train)
        #print(yhat.shape)

        # start = time.perf_counter()
        # bnn.mle_fit(x_train, y_train, epochs=1000, lr=0.001, batch_size=100)
        # end = time.perf_counter()
        # timeused = end - start
        # print(f"{timeused=} seconds of mle fit")

        start = time.perf_counter()
        chain = bnn.hmc_chain(
            x=x_train,
            y=y_train,
            num_results_per_batch=num_results_per_batch,
            num_burnin_steps=num_burnin_steps,
            num_leapfrog_steps=num_leapfrog_steps,
            step_size=step_size,
            adaptation=None,
        )

        # chain = bnn.no_u_turn_chain(
        #     x=x_train,
        #     y=y_train,
        #     num_results_per_batch=num_results_per_batch,
        #     num_burnin_steps=num_burnin_steps,
        #     step_size=step_size,
        #     adaptation=None,
        # )

        # end = time.perf_counter()
        # timeused = end - start
        # print(f"{timeused=} seconds of hmc sampling")

        # # test the model
        # n_test = 1000
        # x_test = tf.random.normal(shape=(n_test, 1), mean=0.0, stddev=3.0)
        # # predictions = bnn(x_test)
        # # print(predictions.shape)
        # predictions = bnn.predict_from_chain(x_test)
        # print(f"{x_test.shape=}")
        # print(f"{predictions.shape=}")

        # x = np.array(list(x_test.numpy().squeeze(-1)) * num_results)
        # predictions = np.array(predictions)
        # predictions = predictions.squeeze(-1).ravel()
        # print(f"{x.shape=}")
        # print(f"{predictions.shape=}")
        # sns.lineplot(x, predictions, ci="sd")
        # # plt.scatter(x_test, predictions, label="model", color="r")

        # x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
        # plt.plot(x, f(x), label="True function", color="r")
        # plt.scatter(x_train, y_train, label="observed data")
        # plt.legend()
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.show()
