import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data
import sys
import time


def main():
    particle_ids = ["1000022"] * 2
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
    x_test, y_test = data.get("test")
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = y_test.squeeze(-1)
    # y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


    # model_names = [
    #     r"models/kernel_nuts_results_1000_burnin_4096_epochs_0_leapfrogsteps_512_nodes_[5, 20, 20, 1].npz",
    #     r"models/kernel_nuts_results_1000_burnin_4096_epochs_2500_leapfrogsteps_512_nodes_[5, 20, 20, 1].npz",
    # ]

    model_names = [
        r"models/kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 2048, 1].npz"
    ]

    # model_names = [
    #     r"old_models/4_small_hidden_layers.npz"
    # ]

    # model_names = [
    #     r"old_models/5_hidden_layers_tanh.npz"
    # ]

    model = BayesianNeuralNetwork()
    model.load_model(fname=r"models/kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 2048, 1].npz")
    # Checks size of the model in GB.
    sz = np.sum(
        [sys.getsizeof(w.numpy()) for w in model.weights]
    ) / 1e9
    print(f"Memory footprint = {sz} GB")
    # model.save_model(fname=r"new_models/kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 2048, 1].npz")
    sys.exit()
    models = [BayesianNeuralNetwork() for _ in model_names]

    for bnn, model_name in zip(models, model_names):
        bnn.load_model(fname=model_name)
    print(*models)
    print([[w.shape for w in bnn.weights] for bnn in models])


    x_test, y_test = data.get("test")
    y_test = y_test.squeeze(-1)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    model_predictions = [bnn(x_test).numpy().squeeze(-1) for bnn in models]
    mean_predictions = [np.mean(p, axis=0) for p in model_predictions]
    mean_std = [np.std(p, axis=0) for p in model_predictions]
    standardized_residuals = [
        (y_test - y_pred) / std for y_pred, std in zip(mean_predictions, mean_std)
    ]
    rel_err = [
        (y_test - y_pred) / y_test for y_pred in mean_predictions
    ]
    rel_err = [err[err >= -1] for err in rel_err]
    rel_err = [err[err <= 1] for err in rel_err]
    n_bins = 100
    plt.hist(standardized_residuals[0], histtype="step", density=True, bins=n_bins)
    # Create normal distribution
    x = np.linspace(min(standardized_residuals[0]), max(standardized_residuals[0]), 1001)
    # x = np.linspace(-2, 2, 1000)
    normal_dist = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    plt.plot(x, normal_dist, color="black", linestyle="--")
    # sns.histplot(standardized_residuals[1], fill=False, )
    plt.show()

    plt.hist(rel_err[0], histtype="step", density=True, bins=n_bins)
    # sns.histplot(standardized_residuals[1], fill=False, )
    plt.show()


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()