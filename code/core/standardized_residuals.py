import re
import sys
import time
import matplotlib.pyplot as plt
plt.rc("text", usetex = True)

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data



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

    # model_names = [
    #     r"models/kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 256, 1].npz"
    # ]

    # model_names = [
    #     r"models/kernel_hmc_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 8192, 1].npz"
    # ]

    # model_names = [
    #     r"old_models/4_small_hidden_layers.npz"
    # ]

    # model_names = [
    #     r"old_models/3_hidden_layers_tanh.npz"
    # ]

    # model_names1 = [
    #     f"models/{i}_small_hidden_layers.npz" for i in range(1, 6)
    # ]

    # model_names2 = [
    #     f"models/{i}_hidden_layers_tanh.npz" for i in range(1, 6)
    # ]

    # model_names = []
    # for name1, name2 in zip(model_names1, model_names2):
    #     model_names.extend([name1, name2])

    model_names = [
        f"models/{i}_hidden_layers_tanh.npz" for i in range(1, 6)
    ]
    

    # model_names = [
    #     f"old_models/{i+1}_small_hidden_layers.npz" for i in range(4)
    # ]

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
    predictions_std = [np.std(p, axis=0) for p in model_predictions]
    num_params = [bnn.num_params for bnn in models]



    standardized_residuals = [
        (y_test - y_pred) / std for y_pred, std in zip(mean_predictions, predictions_std)
    ]
    print(f"{np.mean(standardized_residuals, axis=1)=}")
    print(f"{np.std(standardized_residuals, axis=1)=}")


    rel_err = [
        (y_test - y_pred) / y_test for y_pred in mean_predictions
    ]
    rel_err = [err[err >= -1] for err in rel_err]
    rel_err = [err[err <= 1] for err in rel_err]
    n_bins = 100
    max_x = -100
    min_x = 100

    for i, residual in enumerate(standardized_residuals):
        label = f"Model {i+1}"
        plt.hist(residual, histtype="step", density=True, bins=n_bins, label=label)
        max_x = max(residual) if max(residual) > max_x else max_x
        min_x = min(residual) if min(residual) < min_x else min_x
    
    
        
    # Create normal distribution
    min_x = -5
    max_x = 5   
    x = np.linspace(min_x, max_x, 1001)
    # x = np.linspace(-2, 2, 1000)
    normal_dist = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    plt.plot(x, normal_dist, color="black", linestyle="--", label="$\mathcal{N}(0, 1)$")
    sz = 14
    plt.xticks(size=sz)
    plt.yticks(size=sz)
    plt.xlabel("Standardized residual", fontsize=sz)
    plt.ylabel("Density", fontsize=sz)
    plt.xlim((min_x, max_x))

    plt.legend(fontsize=sz)
    # plt.show()
    path = r"/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/standardized_residuals"
    figname = "/".join([
        path,
        "standardized_residual_simple_models.pdf",
    ])
    # plt.savefig(figname)
    # sns.histplot(standardized_residuals[1], fill=False, )
    plt.show()


    # In target space

    # Propagate error
    predictions_std = [
        10 ** y * np.log(10) * std for y, std in zip(mean_predictions, predictions_std)
    ]
    mean_predictions = [10 ** y for y in mean_predictions]
    y_test = 10 ** y_test




    standardized_residuals = [
        (y_test - y_pred) / std for y_pred, std in zip(mean_predictions, predictions_std)
    ]

    print(f"{np.mean(standardized_residuals, axis=1)=}")
    print(f"{np.std(standardized_residuals, axis=1)=}")


    rel_err = [
        (y_test - y_pred) / y_test for y_pred in mean_predictions
    ]
    rel_err = [err[err >= -1] for err in rel_err]
    rel_err = [err[err <= 1] for err in rel_err]
    n_bins = 100
    max_x = -100
    min_x = 100

    for i, residual in enumerate(standardized_residuals):
        label = f"Model {i+1}"
        plt.hist(residual, histtype="step", density=True, bins=n_bins, label=label)
        max_x = max(residual) if max(residual) > max_x else max_x
        min_x = min(residual) if min(residual) < min_x else min_x

    # Create normal distribution
    min_x = -5
    max_x = 5   
    x = np.linspace(min_x, max_x, 1001)
    # x = np.linspace(-2, 2, 1000)
    normal_dist = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    plt.plot(x, normal_dist, color="black", linestyle="--", label="$\mathcal{N}(0, 1)$")
    sz = 14
    plt.xticks(size=sz)
    plt.yticks(size=sz)
    plt.xlabel("Standardized residual", fontsize=sz)
    plt.ylabel("Density", fontsize=sz)
    plt.xlim((min_x, max_x))

    plt.legend(fontsize=sz)

    path = r"/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/standardized_residuals"
    figname = "/".join([
        path,
        "standardized_residual_target_space.pdf",
    ])
    # plt.savefig(figname)
    plt.show()

if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()
