import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import tqdm
import time


from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHAloader
from utils.preprocessing import split_data


def main_train():
    particle_ids = ["1000022"] * 2
    target_dir = "./targets"
    feat_dir = "./features"
    dl = SLHAloader(
        particle_ids=particle_ids,
        feat_dir=feat_dir,
        target_dir=target_dir,
        target_keys=["nlo"],
    )
    features = dl.features.to_numpy()
    targets = dl.targets.get("nlo").to_numpy()
    targets = np.log10(targets) #Transform targets to log10 space.
    nan_idx = np.isnan(targets)
    idx = (nan_idx == False)
    features = features[idx]
    targets = targets[idx]

    data = split_data(features=features, targets=targets)
    x_train, y_train = data["train"]



    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_train = y_train[:, None]
    input_size = x_train.shape[-1]
    layers = [input_size, 10, 1]


    num_chains = 10
    num_results = 100
    num_burnin_steps = 100
    bnn = BayesianNeuralNetwork(
        layers=layers, activation="tanh", num_chains=num_chains
    )

    bnn.mle_fit(
        x=x_train,
        y=y_train,
        epochs=1000,
        lr=0.001
    )

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        num_leapfrog_steps=60,
        step_size=0.01,
        target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train)
    )

    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps)
    )

    start = time.perf_counter()
    chain = bnn.sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=0,
        fname="models/bnn_mssm_test.npz"
    )
    end = time.perf_counter()




if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main_train()
