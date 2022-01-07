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

def dense(x, w, b, activation):
    return activation(tf.matmul(x, w) + b)

def get_model(weights):
    def model(x):
        kernel = weights[::2]
        bias = weights[1::2]
        for w, b in zip(kernel[:-1], bias[:-1]):
            x = dense(x, w, b, tf.nn.sigmoid)
        x = dense(x, kernel[-1], bias[-1], tf.identity)
        return x
    return model

def log_likelihood(x, y, weights):
    model = get_model(weights)
    yhat = model(x)
    return -0.5 * tf.reduce_sum((y - yhat) ** 2)

def log_prior(weights, lamb=1e-3):
    return lamb * tf.reduce_sum(
        [tf.reduce_sum(w ** 2) for w in weights]
    )

def get_target_log_prob_fn(x, y):
    def target_log_prob_fn(*weights):
        return log_prior(weights) + log_likelihood(x, y, weights)
    return target_log_prob_fn


def get_weights(layers, prior):
    weights = []
    for n, m in zip(layers[:-1], layers[1:]):
        kernel = prior.sample((n, m))
        bias = prior.sample(m)
        weights.extend([kernel, bias])
    return weights

@tf.function
def sample_chain(*args, **kwargs):
    return tfp.mcmc.sample_chain(*args, **kwargs)


def main():
    num_results = 1000
    num_burnin_steps = 0
    num_leapfrog_steps = 60
    step_size = 0.01

    prior = tfp.distributions.Normal(loc=0.0, scale=1.0)
    layers = [1, 50, 1]
    num_chains = 4
    # weights = []
    # for _ in range(num_chains):
    #     weights.append(get_weights(layers, prior))
    weights = get_weights(layers, prior)
    # print(weights)

    num_train = 1000
    x = tf.random.normal(shape=(num_train, 1), mean=0., stddev=2.)
    y = tf.math.sin(x) * tf.math.cos(x)
        
    target_log_prob_fn = get_target_log_prob_fn(x, y)
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
        target_log_prob_fn=target_log_prob_fn
    )

    start = time.perf_counter()
    chain = sample_chain(
        kernel=kernel,
        num_results=num_results,
        current_state=weights,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=0,
        trace_fn=None,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds")



if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()