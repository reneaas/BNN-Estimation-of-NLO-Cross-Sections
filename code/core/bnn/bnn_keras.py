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


def get_model(layers=None, activation=tf.nn.sigmoid):
    if layers is not None:
        input_size = layers[0]
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(units=layers[0], input_shape=(input_size,), activation=activation)
        )
        for units in layers[2:-1]:
            model.add(
                tf.keras.layers.Dense(units=units, activation=activation)
            )
        
        model.add(
            tf.keras.layers.Dense(units=layers[-1], activation=None)
        )
    else:
        model = tf.keras.Sequential()
    
    return model

# @tf.function
def log_prior(current_state, lamb=1e-3):
    res = 0
    for state in current_state:
        res -= tf.reduce_sum(state ** 2)
    return res

def log_likelihood(x, y, current_state, model):
    model.set_weights(current_state)
    y_pred = model(x)
    return -tf.reduce_sum((y_pred - y) ** 2)

def get_target_log_prob_fn(x, y, model):
    def target_log_prob_fn(*current_state):
        return log_prior(current_state) + log_likelihood(x, y, current_state, model)
    return target_log_prob_fn

def main():
    layers = [1, 50, 1]
    model = get_model(layers=layers, activation=tf.nn.sigmoid)
    current_state = model.get_weights()

    num_train = 1000
    x = tf.random.normal(shape=(num_train, 1))
    y = x * tf.math.sin(x) * tf.math.cos(x)

    num_leapfrog_steps = 60
    step_size=0.01
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=get_target_log_prob_fn(x, y, model),
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
    )
    num_results = 1000
    num_burnin_steps = 0

    start = time.perf_counter()
    chain = tfp.mcmc.sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        trace_fn=None,
        current_state=model.get_weights()
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds")


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()