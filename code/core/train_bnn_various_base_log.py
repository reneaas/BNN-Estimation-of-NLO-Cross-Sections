import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_probability as tfp
import pandas as pd
import time
from tqdm import trange
import re
import math

from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data
from utils.trace_functions import (
    trace_fn_hmc,
    trace_fn_nuts,
    trace_fn_adaptive_hmc,
    trace_fn_adaptive_nuts,
)
from utils.metrics import r2_score
import sys 

def load_dataset(particle_ids):
    dl = SLHALoader(
        particle_ids=particle_ids,
        feat_dir="./features",
        target_dir="./targets",
    )

    features = dl.features.to_numpy()
    targets = dl.targets.get("nlo").to_numpy()
    idx = (targets > 0)
    targets = np.exp(targets[idx])
    targets = targets[:, None]
    features = features[idx]

    data = split_data(features=features, targets=targets)
    return data



def main():
    num_results = 1024
    num_steps_between_results = 0
    num_epochs = 1024
    batch_size = 32
    num_leapfrog_steps = 512
    kernel = "hmc"
    num_burnin_steps = 1024
    layers = [5, 20, 20, 1]
    activation = "tanh"
    adaptation_steps = int(0.8 * num_burnin_steps)

    bnn = BayesianNeuralNetwork(
        layers=layers,
        activations="tanh",
        lamb=1e-3,
        likelihood_noise=1.,
    )

    data = load_dataset(particle_ids=["1000022"] * 2)
    x_train, y_train = data.get("train")
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    start = time.perf_counter()
    loss = bnn.mle_fit(
        x_train=x_train,
        y_train=y_train,
        epochs=num_epochs,
        lr=0.001,
        batch_size=batch_size,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"timeused = {timeused} on MLE fit.")


    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=[tf.fill(w.shape, 0.001) for w in bnn.weights],
        target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train),
    )

    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=inner_kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=0.65,
    )
    trace_fn = lambda _, pkr: trace_fn_adaptive_hmc(_, pkr)

    start = time.perf_counter()
    chain, trace = bnn.sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=num_steps_between_results,
        fname="./models/exp_trained_model.npz",
        trace_fn=trace_fn,
    )
    end = time.perf_counter()
    timeused = end - start 
    print(f"{timeused = } seconds on HMC sampling")
    accept_ratio = sum(trace.get("is_accepted").numpy()) / num_results
    print(f"accept ratio = {accept_ratio}")



if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()
