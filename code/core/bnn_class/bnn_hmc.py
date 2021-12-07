import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

np.random.seed(1)
tf.random.set_seed(1)


class BayesianNeuralNetworkHMC:
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
    """
    def __init__(
        self,
        layers=None,
        activation=None,
        top_layer_activation=tf.identity,
        kernel_prior=None,
        bias_prior=None,
        prior_mean=0.0,
        prior_stddev=0.01,
        lamb=0.0,
    ):
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

        # Get initial parameters, if layers are provided.
        if layers:
            tmp = [
                (
                    self.kernel_prior.sample(sample_shape=(n, m)),
                    self.bias_prior.sample(sample_shape=(m,)),
                )
                for n, m in zip(layers[:-1], layers[1:])
            ]

            self.weights = []
            for weights in tmp:
                kernel, bias = weights
                self.weights.append(kernel)
                self.weights.append(bias)
            del tmp  # Delete reference to tmp. Served its purpose.

        if activation:
            self.activation = activation
        else:
            self.activation = tf.nn.sigmoid

        self.top_layer_activation = top_layer_activation

    @tf.function
    def prior_log_prob_fn(self, weights):
        """Log prior probability function of the weights of the network.
        Currently set to be exp(-lamb * w ** 2) according to Radfor Neals treatment
        of BNNs. `lamb` is the analogue to a regularization
        hyperparameter in L2-regularization and serves the same purpose for BNNs.

        Args:
            weights (list[tf.Tensor])   :   List containing tensors with kernels and biases.
        """
        log_prob = 0
        for w in weights:
            log_prob -= tf.reduce_sum(w ** 2)
        return self.lamb * log_prob

    def log_likelihood(self, x, y, weights):
        """Computes log likelihood of predicted target y given features `x` and `weights`.

        Args:
            x  (tf.Tensor)              :   Input features of shape [num_points, num_features]
            y   (tf.Tensor)             :   Predicted targets of shape [num_points, num_outputs]
            weights (list[tf.Tensor])   :   list of weights of the network.

        Returns:
            The log likelihood of yhat as the mean value given target y.
            Equivalent to the residual sum of squares (RSS).
        """
        model = self.build_model(weights)
        yhat = model(x)
        return -0.5 * tf.reduce_sum((y - yhat) ** 2)

    def create_log_prob_fn(self, x, y):
        """Returns the combines log probability function needed to
        use tf.mcmc.sample_chain.

        Args:
            x (tf.Tensor)   :   Input features of shape [num_points, num_features].
                                Usually only used with training data.
            y (tf.Tensor)   :   Predicted targets of shape [num_points, num_outputs].
                                Usually only used with training data.

        Returns:
            target_log_prob_fn (target log probability function used in the MCMC chain).
        """

        def target_log_prob_fn(*weights):
            log_prob = self.prior_log_prob_fn(weights) + self.log_likelihood(
                x, y, weights
            )
            return log_prob

        return target_log_prob_fn

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
                x = self.activation(tf.matmul(x, w) + b)
            x = self.top_layer_activation(tf.matmul(x, kernel[-1]) + bias[-1])
            return x

        return model

    def predict_from_chain(self, x, chain=None):
        predictions = []
        if chain:
            for weights in chain:
                model = self.build_model(weights)
                predictions.append(model(x).numpy())
        else:
            for weights in self.chain:
                model = self.build_model(weights)
                predictions.append(model(x).numpy())
        return predictions

    @tf.function
    def sample_chain(self, *args, **kwargs):
        return tfp.mcmc.sample_chain(*args, **kwargs)

    def hmc_chain(
        self, x, y, num_results, num_burnin_steps, num_leapfrog_steps, step_size
    ):
        current_state = self.weights
        target_log_prob_fn = self.create_log_prob_fn(x, y)
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            store_parameters_in_results=None,
        )

        self.chain = self.sample_chain(
            kernel=kernel,
            current_state=current_state,
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=0,
            trace_fn=None,
        )

        self.chain = [
            [tensor[i] for tensor in self.chain] for i in range(len(self.chain[0]))
        ]

        return self.chain


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        layers = [1, 50, 1]

        # Create training data
        n_train = 5000
        dims = 1
        f = lambda x: tf.math.sin(x)
        x_train = tf.random.normal(shape=(n_train, dims), mean=0.0, stddev=3.0)
        y_train = f(x_train)

        # Should not be needed.
        # kernel_prior = tfp.distributions.Normal(loc=0., scale=0.01)
        # bias_prior = tfp.distributions.Normal(loc=0., scale=0.01)

        num_results = 1000
        num_burnin_steps = 1000
        num_leapfrog_steps = 60
        step_size = 0.001
        bnn = BayesianNeuralNetworkHMC(layers=layers, activation=tf.nn.sigmoid)

        start = time.perf_counter()
        chain = bnn.hmc_chain(
            x=x_train,
            y=y_train,
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            num_leapfrog_steps=num_leapfrog_steps,
            step_size=step_size,
        )
        end = time.perf_counter()
        timeused = end - start
        print(f"{timeused=} seconds of hmc sampling")

        # test the model
        n_test = 1000
        x_test = tf.random.normal(shape=(n_test, 1), mean=0.0, stddev=3.0)
        predictions = bnn.predict_from_chain(x_test)

        x = np.array(list(x_test.numpy().squeeze(-1)) * num_results)
        predictions = np.array(predictions)
        predictions = predictions.squeeze(-1).ravel()
        sns.lineplot(x, predictions, ci="sd")

        x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
        plt.plot(x, f(x), label="True function", color="r")
        plt.scatter(x_train, y_train, label="observed data")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
