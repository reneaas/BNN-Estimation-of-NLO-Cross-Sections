import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import time
import sys

np.random.seed(10)


from bnn.bnn import BayesianNeuralNetwork
from bnn.bnn_base import _BNNBase
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data
from utils.trace_functions import (
    trace_fn_hmc,
    trace_fn_nuts,
    trace_fn_adaptive_hmc,
    trace_fn_adaptive_nuts,
)


def main():
    feat_dir = "features"
    target_dir = "targets"

    processes = [
        ["1000022"] * 2,
        ["1000023"] * 2,
        ["1000025"] * 2,
        ["1000035"] * 2,
    ]

    data = {"features": [], "targets": []}
    for process in processes:
        dl = SLHALoader(particle_ids=process, feat_dir=feat_dir, target_dir=target_dir)
        features = dl.features.to_numpy()
        targets = dl.targets.get("nlo").to_numpy()
        data["features"].append(features)
        data["targets"].append(targets)
    features = np.concatenate(data["features"], axis=0)
    targets = np.concatenate(data["targets"], axis=0)
    del data

    # Remove ill-defined datapoints.
    idx = (targets > 0) 
    targets = targets[idx]
    features = features[idx]
    targets = np.log10(targets)  # Transform targets to log10 space.
    targets = targets[:, None]

    #Reshuffle data.
    idx = np.random.permutation(features.shape[0])
    features = features[idx]
    targets = targets[idx]

    data = split_data(features=features, targets=targets)
    x_train, y_train = data.get("train")

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    del data

    # Define basic parameters of the BNN

    num_chains = 1
    layers = [5, 20, 20, 20, 20, 1]
    activations = "swish"
    bnn = BayesianNeuralNetwork(
        layers=layers,
        activations=activations,
        likelihood_noise=1.0,
        num_chains=num_chains,
    )

    start = time.perf_counter()
    bnn.mle_fit(
        x_train=x_train,
        y_train=y_train,
        epochs=10000,
        lr=0.001,
        batch_size=32,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"Timeused = {timeused} seconds on MLE fit.")


    num_results = 1000
    num_burnin_steps = 1000

    inner_kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=bnn.get_target_log_prob_fn(x_train, y_train),
        step_size=[0.0001 for _ in bnn.weights],
        max_tree_depth=10,
    )


    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=inner_kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
    )

    start = time.perf_counter()
    chain, trace = bnn.sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=10,
        fname="models/data_multiplication_model_10000_epochs_pretraining.npz",
        trace_fn=trace_fn_adaptive_nuts,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"timeused = {timeused} seconds on MCMC sampling.")

    accept_ratio = sum(trace.get("is_accepted").numpy()) / num_results
    print(f"accept ratio = {accept_ratio}")
    if isinstance(inner_kernel, tfp.mcmc.NoUTurnSampler):
        print("Mean number of leapfrog steps taken: ", tf.reduce_mean(trace["leapfrogs_taken"]).numpy())



if __name__ == "__main__":
    with tf.device("/GPU:0"):
        main()
