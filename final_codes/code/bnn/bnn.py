from typing import Callable, Optional, Union, List
import time
import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import seaborn as sns

if __name__  == "__main__":
    from bnn_base import _BNNBase
else:
    from .bnn_base import _BNNBase

np.random.seed(10)
tf.random.set_seed(10)


class BayesianNeuralNetwork(_BNNBase):
    """Class for a Bayesian neural network (BNN) using Hamiltonian Monte Carlo (HMC)
    and its derivatives as a sampling method to sample from its posterier.

    Attributes:
        lamb (float):
            Regularization parameter.
        avail_activations (dict[callable]):
            Dictionary with available activation functions.
        activation (list[callable]):
            List of python callable activation functions.
        weights (list[tf.Tensor]):
            List of the weights of the neural network.
        _num_chains (int):
            Number of Monte Carlo chains run in parallel.
        kernel_prior (tfp.distributions.Distribution):
            Kernel prior distribution.
        bias_prior (tfp.distributions.Distribution):
            Bias prior distribution.
        num_layers (int):
            Number of layers.

    #### Examples

    ```python
    num_chains = 100
    num_results = 100
    num_burnin_steps = 1000


    input_size = 5
    output_size = 1
    layers = [input_size, 10, 20, 10, output_size]
    activation = ["relu", "tanh", "leaky_relu", "identity"]
    bnn = BayesianNeuralNetwork(
        layers=layers, activation=activation, num_chains=num_chains,
    )

    #  Pretrain the network layers before running MCMC chain.
    loss = bnn.mle_fit(
        x=x_train,
        y=y_train,
        epochs=10000,
        batch_size=None,
    )

    # Create tfp.mcmc.TransitionKernel
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=bnn.get_target_log_prob_fn(x_train, y_train),
        num_leapfrog_steps=100,
        step_size=0.01
    )

    # Create adaptive transition kernel with HMC as the inner kernel.
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps)
    )

    # Run MCMC chain.
    chain = bnn.sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=0,
        fname="my_model.npz",
        trace_fn=None,
    )

    ```

    """

    def __init__(
        self,
        layers=None,
        activations=None,
        kernel_prior=None,
        bias_prior=None,
        lamb=1e-3,
        likelihood_noise=0.1,
        num_chains=1,
    ):
        """Creates the bayesian neural network.

        Args:
            layers (list, optional):
                List containing the nodes of each layer.
                Shape is [input_size, nodes_of_layer1, ..., nodes_of_layerN, num_outputs]
            activation (list[str, function], optional):
                List of activation functions, one per each layer. len(activation) = len(layers) - 1.
                Available activations: `sigmoid`, `relu`, `leaky_relu`, `tanh`, `identity`.
                User can also provide their tf.nn callable equivalents, such as `tf.nn.sigmoid`.
                It's recommended to provide the activations' name, not its tf.nn equivalent.
            kernel_prior (tfp.distributions.Distribution):
                If set to None, it defaults to tfp.distributions.Normal
            bias_prior (tfp.distributions.Distribution):
                If set to None, it defaults to tfp.distributions.Normal
            prior_mean (float):
                Mean value of the tfp.distribution.Normal. Default: `prior_mean = 0.0`
            prior_stddev (float):
                Standard deviation of tfp.distribution.Normal. Default: `prior_stddev = 0.01`
            lamb (float):
                Regularization parameter. Default `prior_noise=1e-3`.
            likelihood_noise (float):
                Gaussian likelihood noise. Default: `likelihood_noise=1e-3`.
            num_chains (int):
                Number of chains of the parameters to be sampled using MCMC methods.
                If `num_chains` > 1, the MCMC sampling chain will run multiple chains in parallel.
                Default: `num_chains=1`.

        Raises:
            ValueError:
                - If len(activation) == len(layers) - 1 is False.
                - If activations is a list, and an element is str, but is not an available activation function.
            TypeError: 
                if activations is a list, but the elements is not str nor a Python callable.
        """
        super().__init__(
            layers=layers,
            activations=activations,
            kernel_prior=kernel_prior,
            bias_prior=bias_prior,
            num_chains=num_chains,
        )
        self.lamb = lamb
        self.likelihood_noise = likelihood_noise


    @tf.function
    def __call__(self, x, weights=None):
        """Performs a simple forward pass with set of weights and a given input x

        Args:
            x (tf.Tensor):
                Input features of shape (num_points, num_features)
            weights (optional, list[tf.Tensor]):
                List of weights of the network of shape (batch_size, n, m).
                Default: `weights=None`. In this case, it defaults to the weights stored in the class.

        Returns:
            The computed outputs of the forward pass.
            A tf.Tensor object of shape [num_weights, num_points, num_outputs]

        """
        if weights is None:
            kernel = self._weights[::2]
            bias = self._weights[1::2]
        else:
            kernel = weights[::2]
            bias = weights[1::2]

        for w, b, activation in zip(kernel, bias, self._activations):
            x = self._dense_layer(x, w, b, activation)
        return x

    @tf.function
    def loss_fn(self, y_pred, y_true, weights):
        return (
            0.5 * tf.reduce_sum((y_pred - y_true) ** 2, axis=(-1, -2))
            - self._log_prior_prob_fn(weights)
        )

    @tf.function
    def _train_step(self, x, y):
        """Computes a training step in conjunction with the MLE fit
        method.

        Args:
            x (tf.Tensor):
                Input features of shape [num_points, num_features]
            y (tf.Tensor):
                Targets of shape [num_points, num_outputs]

        Returns:
            loss (tf.Tensor):
                Loss of shape=().
        """
        with tf.GradientTape() as tape:
            tape.watch(self._weights)
            y_pred = self(x, self._weights)
            loss = self.loss_fn(y_pred=y_pred, y_true=y, weights=self._weights)
        grad = tape.gradient(loss, self._weights)
        self.optimizer.apply_gradients(zip(grad, self._weights))
        return loss

    def mle_fit(
        self,
        x_train,
        y_train,
        epochs,
        lr,
        batch_size=None,
        optimizer=None,
        x_val=None,
        y_val=None,
        dont_break=False,
    ):
        """Fits the network using the backpropagation algorithm, i.e maximum likelihood estimation (MLE).

        Args:
            x_train (tf.Tensor):
                Training features of shape [num_points, num_features]
            y_train (tf.Tensor):
                Training targets of shape [num_points, num_outputs]
            epochs (int):
                Number of epochs to fit the network.
            lr (float):
                Learning rate.
            batch_size (int, optional):
                Batch size used during training.
            optimizer(tf.keras.optimizers.Optimizer, optional):
                Optimizer used during training.

        Returns:
            loss (list[float]):
                List containing the loss per epoch.
        """
        if len(x_train.shape) != 2:
            raise ValueError(
                f"""
                len(x.shape) = {len(x_train.shape)} != 2. The shape of the input features must be (num_points, num_features)
            """
            )
        if len(y_train.shape) != 2:
            raise ValueError(
                f"""
                len(y.shape) = {len(y_train.shape)} != 2. The shape of the input targets must be (num_points, num_targets)
            """
            )

        #Validation loss is computed to facilitate early stopping.
        validate = False
        if x_val is not None and y_val is not None:
            validate = True
            val_losses = []

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
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
        
        if validate:
            val_loss = 2 ** 30
        train_losses = []
        self._weights = [tf.Variable(w) for w in self._weights]
        if batch_size is not None and x_train.shape[0] > batch_size:
            ds = self._get_dataset(x=x_train, y=y_train, batch_size=batch_size)
            for _ in trange(epochs, desc="Epochs"):
                tot_loss = 0
                for x_, y_ in ds:
                    loss = self._train_step(x_, y_)
                    tot_loss += loss
                train_losses.append(tot_loss)
        else:
            for _ in trange(epochs, desc="Epochs"):
                loss = self._train_step(x_train, y_train)
                train_losses.append(loss)

                if validate:
                    y_pred = self(x_val, self._weights)
                    tmp = self.loss_fn(y_pred=y_pred, y_true=y_val, weights=self._weights)
                    val_losses.append(tmp)
                    if tmp > val_loss:
                        if dont_break is False:
                            break
                    else:
                        val_loss = tmp
        losses = {"train_loss": train_losses}
        if validate:
            losses["val_loss"] = val_losses

        return losses

    @tf.function
    def _log_prior_prob_fn(self, weights):
        """Log prior probability function of the weights of the network.
        Currently set to be exp(-lamb * w ** 2) according to Radford Neals treatment
        of BNNs. `lamb` is the analogue to a regularization
        hyperparameter in L2-regularization and serves the same purpose for BNNs.

        Args:
            weights (list[tf.Tensor]):
                List containing tensors with kernels and biases.

        Returns:
            tf.Tensor with the computed prior log probability. Returned shape: [num_chains]
        """
        kernel = weights[::2]
        bias = weights[1::2]
        res = 0
        for w, b in zip(kernel, bias):
            res += tf.reduce_sum(w ** 2, axis=(-1, -2))
            res += tf.reduce_sum(b ** 2, axis=-1)
        return -0.5 * self.lamb * res

    @tf.function
    def _log_likelihood_fn(self, x, y, weights):
        """Computes log likelihood of predicted target y given features `x` and `weights`.

        Args:
            x (tf.Tensor):
                Input features of shape (num_points, num_features)
            y (tf.Tensor):
                Predicted targets of shape (num_points, num_outputs)
            weights (list[tf.Tensor]):
                list of weights of the network.

        Returns:
            The log likelihood of yhat as the mean value given target y.
            Equivalent to the residual sum of squares (RSS). Returned shape [num_chains]
        """
        yhat = self(x, weights)
        return (
            -0.5 * self.likelihood_noise * tf.reduce_sum((y - yhat) ** 2, axis=(-1, -2))
        )

    def get_target_log_prob_fn(self, x, y):
        """Returns the target log probability function
        used with tf.mcmc.sample_chain.

        Args:
            x (tf.Tensor):
                Input features of shape (num_points, num_features).
                Usually only used with training data.
            y (tf.Tensor):
                Predicted targets of shape (num_points, num_outputs).
                Usually only used with training data.

        Returns:
            target_log_prob_fn (Callable):
                target log probability function used in the MCMC chain.
        """

        def target_log_prob_fn(*weights):
            return self._log_prior_prob_fn(weights) + self._log_likelihood_fn(
                x, y, weights
            )
        return target_log_prob_fn
    
    @tf.function
    def sample_chain_parallel(
        self,
        kernel,
        num_results,
        num_burnin_steps,
        num_steps_between_results=0,
        trace_fn=None,
        parallel_iterations=10,
        fname=None,
    ):
        """Runs the sample chain over multiple GPUs"""
        if not isinstance(kernel, tfp.mcmc.TransitionKernel):
            raise TypeError(
                f"""kernel = {kernel} is not a valid kernel. Please provide an
                instance of tfp.mcmc.TransitionKernel.
                """
            )
        physical_devices = tf.config.list_physical_devices("GPU")
        print(f"Physical devices = {physical_devices}")
        logical_devices = tf.config.list_logical_devices("GPU")
        print(f"Logical devices = {logical_devices}")

        strategy = tf.distribute.MirroredStrategy()
        device_chains = strategy.run(
            tfp.mcmc.sample_chain,
            kwargs={
                "num_results": num_results,
                "current_state": self._weights,
                "kernel": kernel,
                "num_burnin_steps": num_burnin_steps,
                "trace_fn": None,
                "parallel_iterations": parallel_iterations,
                "num_steps_between_results": num_steps_between_results,
            }
        )
        return device_chains

    @tf.function
    def sample_chain_verbose(
        self,
        kernel,
        num_results,
        num_burnin_steps,
        num_steps_between_results=0,
        trace_fn=0,
        parallel_iterations=10,
        fname=None,
    ):
        
        current_state = self.weights
        kernel_results = kernel.bootstrap_results(current_state)

        #Burn in steps:
        for i in trange(num_burnin_steps, desc="Burn-in steps"):
            current_state, kernel_results = kernel.one_step(
                current_state, kernel_results,
            )
        
        chain = current_state

        for i in trange(num_results, desc="Sampling results"):
            current_state, kernel_results = kernel.one_step(
                current_state, kernel_results,
            )
            chain = [tf.concat([state1, state2], axis=0) for state1, state2 in zip(chain, current_state)]
            #chain.append(current_state)
        
        self.chain = chain
            
        
        


    def sample_chain(
        self,
        kernel,
        num_results,
        num_burnin_steps,
        num_steps_between_results=0,
        trace_fn=None,
        parallel_iterations=10,
        fname=None,
        restack=True,
    ):
        """Produces a MCMC chain given a transition kernel.

        Args:
            kernel (tfp.mcmc.TransitionKernel):
                Transition kernel to produce MCMC samples.
            num_results (int):
                number of results per chain.
            num_burnin_steps (int):
                Number of burn-in steps per chain
            num_steps_between_results (int):
                Number of steps in between results. Thinning.
            trace_fn (callable):
                Python callable. Trace function that traces the MCMC chain.
            parallel_iterations (int):
                Number of parallel iterations run on hardware supporting SIMD instructions.
            fname (str):
                Filename to save the sample chain to.

        Returns:
            self.chain (list[tf.Tensor]):
                List of MCMC chain.
            trace (collections.namedtuple, optional):
                Returned if `trace_fn` is not None.

        Raises:
            TypeError if `kernel` is not of type tfp.mcmc.TransitionKernel.
        """
        if not isinstance(kernel, tfp.mcmc.TransitionKernel):
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
            current_state=self._weights,
        )
        if trace_fn is not None:
            chain, trace = chain

        # Restack the chain and assign them as the weights of the model.
        if restack is True:
            self._weights = self._restack_chain(chain)
        else:
            self._weights = chain
        if fname is not None:
            self.save_model(fname)

        if trace_fn is not None:
            return self._weights, trace
        else:
            return self._weights



def trace_fn_adaptive_no_u_turn(_, pkr):
    return {
        "target_log_prob": pkr.inner_results.target_log_prob,
        "leapfrogs_taken": pkr.inner_results.leapfrogs_taken,
        "has_divergence": pkr.inner_results.has_divergence,
        "energy": pkr.inner_results.energy,
        "log_accept_ratio": pkr.inner_results.log_accept_ratio,
        "is_accepted": pkr.inner_results.is_accepted,
        "step_size": pkr.inner_results.step_size,
        "target_accept_prob": pkr.target_accept_prob,
    }


def trace_fn_adaptive_hmc(_, pkr):
    return {
        # "target_log_prob": pkr.inner_results.target_log_prob,
        # "leapfrogs_taken": pkr.inner_results.leapfrogs_taken,
        # "has_divergence": pkr.inner_results.has_divergence,
        # "energy": pkr.inner_results.energy,
        # "log_accept_ratio": pkr.inner_results.log_accept_ratio,
        "is_accepted": pkr.inner_results.is_accepted,
        "step_size": pkr.inner_results.accepted_results.step_size,
        "target_accept_prob": pkr.target_accept_prob,
    }

def trace_fn_chees(_, pkr):
  return (
        pkr.inner_results.accepted_results,
        pkr.step,
        pkr.max_trajectory_length,
        pkr.inner_results.log_accept_ratio,
        pkr.adaptation_rate,
        pkr.averaged_max_trajectory_length,
  )


def main():
    layers = [1, 20, 20, 1]
    n_train = 1000
    n_dims = 1
    f = lambda x: x * tf.math.sin(x) * tf.math.cos(x)
    x_train = tf.random.normal(shape=(n_train, n_dims), mean=0.0, stddev=3.0)
    y_train = f(x_train) + tf.random.normal(shape=x_train.shape, mean=0.0, stddev=0.5)

    num_chains = 1
    num_results = 100
    tot_num_results = num_results * num_chains
    num_burnin_steps = 100
    num_leapfrog_steps = 100
    step_size = 0.0001
    activations = "tanh"

    bnn = BayesianNeuralNetwork(
        layers=layers,
        activations=activations,
        lamb=1e-3,
        likelihood_noise=1.,
        num_chains=num_chains,
    )


    x_val = tf.random.normal(shape=(1000, 1), mean=0.0, stddev=2.0)
    y_val = f(x_val)

    start = time.perf_counter()
    loss = bnn.mle_fit(
        x_train=x_train,
        y_train=y_train,
        epochs=1000,
        lr=0.001,
        batch_size=None,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds on MLE fit.")
    
    if loss.get("val_loss") is not None:
        plt.plot(loss.get("val_loss"))
        plt.xlabel("epochs")
        plt.ylabel("validation loss")
        plt.show()

    # step_size = tf.fill((num_chains, 1), 0.01)
    step_size = [tf.fill(w.shape, 0.001) for w in bnn.weights]
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=bnn.get_target_log_prob_fn(x_train, y_train),
        num_leapfrog_steps=500,
        step_size=step_size,
    )
    # Adaptive kernelx
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=inner_kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=0.65,
    )


    start = time.perf_counter()
    chain = bnn.sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=0,
        fname=None,
        trace_fn=None,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} on sampling")

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
        alpha=0.5,
    )
    plt.plot(
        x_test.numpy().squeeze(1),
        f(x_test).numpy().squeeze(-1),
        label="ground truth",
        color="g",
    )
    plt.legend()
    plt.show()

    weights = bnn.weights
    for i in range(3):
        print(weights[0].shape)
        sns.kdeplot(weights[0][:, 0, i])
        plt.figure()
    plt.show()


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()
