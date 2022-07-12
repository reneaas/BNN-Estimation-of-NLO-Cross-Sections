import tensorflow as tf
import tensorflow_probability as tfp 
import numpy as np
import time
import sys

@tf.function
def forward(x, weights, activations):
    """Calculates the forward pass of the neural network

    Args:
        x (tf.Tensor with dtype tf.float32):
            Input features of shape (num_points, num_features)
        weights (List[tf.Tensor]):
            List containing the weights of the neural network. 
            Assumed structure: weights = [kernel_0, bias_0, kernel_1, bias_1, ..., kernel_L, bias_L]
        activations (List[Callable]):
            List of activation functions. Must be Python callables.

    Returns:
        The result of the forward pass. Shape (num_chains, num_points, num_outputs)   
    """
    kernel = weights[::2]
    bias = weights[1::2]
    for w, b, activation in zip(kernel, bias, activations):
        x = activation(tf.matmul(x, w) + b[..., None, :])
    return x

def get_target_log_prob_fn(x, y, activations):
    """ Returns the target log probability function of the neural network model.

    Args:
        x (tf.Tensor):
            Input features of shape (num_points, num_features)
        y (tf.Tensor):
            Targets of shape (num_points, num_outputs)
        activations (List[Callable]):
            List of activation function used with in the neural network.
            Must consist of Python callables.

    Returns:
        Python Callable with the proper syntax for usage with 
        TensorFlow-Probability's MCMC samplers. 
    
    """
    def log_posterior_fn(*weights):
        return log_prior_fn(weights) + log_likelihood_fn(x, y, weights, activations)
    return log_posterior_fn

@tf.function
def log_prior_fn(weights, lamb=1e-3):
    """Calculates the log prior of the weights of the neural network.
    The prior is assumed to be Gaussian.

    Args:
        weights (List[tf.Tensor]):
            List containing the weights of the neural network. 
            Assumed structure: weights = [kernel_0, bias_0, kernel_1, bias_1, ..., kernel_L, bias_L]
        lamb (float):
            Regularization strength. Default: `lamb=1e-3`.

    Returns:
        The calculated log prior as a tf.Tensor of shape (num_chains,)
    """
    kernel, bias = weights[::2], weights[1::2]
    res = 0
    for w, b in zip(kernel, bias):
        res += tf.reduce_sum(w ** 2, axis=(-1, -2))
        res += tf.reduce_sum(b ** 2, axis=-1)
    return 0.5 * lamb * res

@tf.function
def log_likelihood_fn(x, y, weights, activations):
    """Calculates the Log likelihood given a dataset (x, y),
    and the weights and activations of the neural network.

    Args:
        x (tf.Tensor):
            Input features of shape (num_points, num_features)
        y (tf.Tensor):
            Target of shape (num_points, num_outputs)

    Returns:
        The calculacted log likelihood as a tf.Tensor of shape (num_chains,)
    """
    y_pred = forward(x=x, weights=weights, activations=activations)
    return 0.5 * tf.reduce_sum((y_pred - y) ** 2, axis=(-1,-2))

def get_weights(layers, num_chains, mean=0.0, stddev=1.0):
    """Initates the weights of the neural network model from 
    a Gaussian prior.

    Args:
        layers (List[int]): 
            List specifying the nodes per layer. 
            Structure: layers = [num_features, hidden_layer_1, ..., hidden_layer_L, num_outputs]
        num_chains (int):
            Number of independent Markov chains to run with the MCMC sampler.
        mean (float):
            Mean of the Gaussian prior
        stddev (float):
            Standard deviation of the Gaussian prior.
        
    Returns:
        List of weights of the neural network.
        Structure: weights = [kernel_0, bias_0, kernel_1, bias_1, ..., kernel_L, bias_L]
    """
    weights = []
    for n, m in zip(layers[:-1], layers[1:]):
        w = tf.random.normal(shape=(num_chains, n, m), mean=mean, stddev=stddev)
        b = tf.random.normal(shape=(num_chains, m), mean=mean, stddev=stddev)
        weights.extend([w, b])
    return weights

@tf.function
def sample_chain(*args, **kwargs):
    """Wrapper around tfp.mcmc.sample_chain compiled with tf.function
    Yields a significant speedup.
    """
    return tfp.mcmc.sample_chain(*args, **kwargs)



def main():
    n_train = 1000
    x_train = tf.random.normal(shape=(n_train, 1), mean=0.0, stddev=3.0)
    f = lambda x: x * tf.math.cos(x) * tf.math.sin(x)
    y_train = f(x_train)

    layers = [1, 10, 10, 1]
    activations = [tf.nn.swish, tf.nn.swish, tf.identity]
    num_results = 1000
    num_burnin_steps = 1000
    num_chains = 10
    weights = get_weights(layers=layers, num_chains=num_chains)

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        num_leapfrog_steps=100,
        step_size=0.001,
        target_log_prob_fn=get_target_log_prob_fn(x=x_train, y=y_train, activations=activations)
    )

    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps = int(0.8 * num_burnin_steps)
    )

    start = time.perf_counter()
    chain = sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        kernel=kernel,
        trace_fn=None,
        current_state=weights,
    )
    end = time.perf_counter()
    timeused = end - start 
    print("timeused = ", timeused, " seconds")



if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()