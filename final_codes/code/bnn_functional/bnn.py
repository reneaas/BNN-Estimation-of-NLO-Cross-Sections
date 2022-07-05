import tensorflow as tf
import tensorflow_probability as tfp 
import numpy as np
import time

@tf.function
def forward(x, weights, activations):
    kernel = weights[::2]
    bias = weights[1::2]
    for w, b, activation in zip(kernel, bias, activations):
        x = activation(tf.matmul(x, w) + b[..., None, :])
    return x

def target_log_prob_fn(x, y, activations):
    def log_posterior_fn(*weights):
        return log_prior_fn(weights) + log_likelihood_fn(x, y, weights, activations)
    return log_posterior_fn

@tf.function
def log_prior_fn(weights, lamb=1e-3):
    kernel, bias = weights[::2], weights[1::2]
    res = 0
    for w, b in zip(kernel, bias):
        res += tf.reduce_sum(w ** 2, axis=(-1, -2))
        res += tf.reduce_sum(b ** 2, axis=-1)
    return 0.5 * lamb * res

@tf.function
def log_likelihood_fn(x, y, weights, activations):
    y_pred = forward(x=x, weights=weights, activations=activations)
    return 0.5 * tf.reduce_sum((y_pred - y) ** 2, axis=(-1,-2))

def get_weights(layers, num_chains):
    weights = []
    for n, m in zip(layers[:-1], layers[1:]):
        w = tf.random.normal(shape=(num_chains, n, m), mean=0.0, stddev=1.0)
        b = tf.random.normal(shape=(num_chains, m), mean=0.0, stddev=1.0)
        weights.extend([w, b])
    return weights

@tf.function
def sample_chain(*args, **kwargs):
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
    num_chains = 1
    weights = get_weights(layers=layers, num_chains=num_chains)
    

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        num_leapfrog_steps=100,
        step_size=0.001,
        target_log_prob_fn = target_log_prob_fn(x=x_train, y=y_train, activations=activations)
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