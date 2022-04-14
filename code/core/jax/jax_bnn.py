import numpy as np
from functools import partial
import time
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import sys


@partial(jax.jit, static_argnames=("activation"))
def dense(w, b, x, activation):
    return activation(jnp.matmul(x, w) + b[None, :])

@partial(jax.jit, static_argnames=("activation"))
def forward(weights, x, activation=jax.nn.tanh):
    kernel = weights[::2]
    bias = weights[1::2]
    for w, b in zip(kernel[:-1], bias[:-1]):
        x = dense(w, b, x, activation)
    x = dense(kernel[-1], bias[-1], x, activation)
    return x

@partial(jax.jit, static_argnames=("lamb"))
def log_prior(weights, lamb=1e-3):
    res = 0
    kernel = weights[::2]
    bias = weights[1::2]
    for w, b in zip(kernel, bias):
        res += jnp.sum(w ** 2)
        res += jnp.sum(b ** 2)
    return -0.5 * lamb * res

@jax.jit
def log_likelihood(weights, x, y):
    y_pred = forward(weights, x)
    return -0.5 * jnp.sum((y_pred - y) ** 2)

def get_target_log_prob_fn(x, y):
    def target_log_prob_fn(*weights):
        return log_prior(weights) + log_likelihood(weights, x, y)
    return target_log_prob_fn

def get_weights(layers):
    weights = []
    for n, m in zip(layers[:-1], layers[1:]):
        w = np.random.normal(size=(n, m))
        b = np.random.normal(size=(m))
        w = jnp.array(w)
        b = jnp.array(b)
        weights.extend([w, b])
    return weights

# @jax.jit
def sample_chain(*args, **kwargs):
    return tfp.mcmc.sample_chain(*args, **kwargs)

@partial(jax.jit, static_argnames=("target_log_prob_fn"))
def run_chain(key, current_state, target_log_prob_fn):
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        num_leapfrog_steps=500,
        step_size=0.001,
        target_log_prob_fn=target_log_prob_fn,
    )

    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=80,
    )

    chain = tfp.mcmc.sample_chain(
        kernel=kernel,
        current_state=current_state,
        trace_fn=None,
        num_burnin_steps=100,
        num_results=100,
        seed=key,
    )
    return chain


def main():

    layers = [1, 10, 10, 1]
    weights = get_weights(layers=layers)
    
    x_train = jnp.array(np.random.normal(size=(1000, 1)))
    # x_train = np.random.normal(size=1)
    f = lambda x: x * jnp.cos(x) * jnp.sin(x)
    y_train = f(x_train)
    y_pred = forward(weights, x_train)
    print(y_pred.shape)
    # y_pred = dense(weights[0], weights[1], x_train, jax.nn.swish)


    # print(log_prior(weights))
    # print(log_likelihood(weights, x_train, y_train))

    # kernel = tfp.mcmc.HamiltonianMonteCarlo(
    #     num_leapfrog_steps=100,
    #     step_size=0.001,
    #     target_log_prob_fn=get_target_log_prob_fn(x_train, y_train),
    # )

    init_key, sample_key = jax.random.split(jax.random.PRNGKey(0))
    start = time.perf_counter()
    chain = run_chain(key=sample_key, current_state=weights, target_log_prob_fn=get_target_log_prob_fn(x_train, y_train))
    end = time.perf_counter()
    timeused = end - start
    print(f"timeused = {timeused} seconds")
    
    x_test = jnp.array(np.random.normal(size=(1000, 1)))
    # y_pred = forward(weights=chain, x=x_test)
    weight_shapes = [w.shape for w in chain]
    print(weight_shapes)
    y_pred = forward(weights=chain, x=x_test)
    print(y_pred.ravel())
    


    

if __name__ == "__main__":
    main()
