import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import tqdm
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


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
    particle_ids = ["1000022", "1000022"]
    target_dir = "./targets"
    feat_dir = "./features"
    dl = SLHALoader(
        particle_ids=particle_ids,
        feat_dir=feat_dir,
        target_dir=target_dir,
        target_keys=["nlo"],
    )
    features = dl.features.to_numpy()
    targets = dl.targets.get("nlo").to_numpy()
    idx = (targets > 0)
    targets = np.log10(targets[idx])
    targets = targets[:, None]
    features = features[idx]

    data = split_data(features=features, targets=targets)
    x_train, y_train = data["train"]

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    input_size = x_train.shape[-1]
    layers = [input_size, 10, 10, 1]
    activations = "tanh"
    num_burnin_steps = 1024
    num_results = 8192

    bnn = BayesianNeuralNetwork(
        layers=layers,
        activations=activations,
        lamb=1e-3,
        likelihood_noise=1.,
        num_chains=10,
    )

    weights = bnn.weights
    print([p.shape for p in weights])

    start = time.perf_counter()
    loss = bnn.mle_fit(
        x_train=x_train,
        y_train=y_train,
        epochs=4096,
        lr=1e-3,
        batch_size=32,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused = } seconds (on MLE to find point estimate)")

  

    inner_kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train),
        step_size=[tf.fill(w.shape, 0.001) for w in bnn.weights],
        max_tree_depth=10,
    )

    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=inner_kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=0.75,
    )


    fname = "./models/multi_chain_model_hmc4.npz"
    start = time.perf_counter()
    chain = bnn.sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        fname=fname,
        trace_fn=None,
        num_steps_between_results=0,
        restack=False,
    )

    # accept_ratio = sum(trace.get("is_accepted").numpy()) / num_results
    # print(f"{accept_ratio = }")
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused = } on sampling")


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()