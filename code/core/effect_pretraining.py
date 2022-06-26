import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_probability as tfp
import pandas as pd
import time
from tqdm import trange
import re

from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data
from utils.metrics import r2_score
import sys 

# def get_data(root_dir):
#     pattern = r"(\[.+?\]|\d+\.?\d*|\w+)"
#     data = []
#     with os.scandir(root_dir) as iter:
#         for entry in iter:
#             if entry.name.endswith(".txt") and entry.is_file():
#                 with open(entry.path, "r") as infile:
#                     keys = infile.readline()
#                     keys = keys.split()
#                     vals = re.findall(pattern, infile.readline())
#                     tmp_data = {key: val for key, val in zip(keys, vals)}
#                     data.append(tmp_data)
#     return data


def get_data(fname):
    pattern = r"(\[.+?\]|\d+\.?\d*|\w+)"
    with open(fname, "r") as infile:
        keys = infile.readline()
        keys = keys.split()
        vals = re.findall(pattern, infile.readline())
        data = {key: val for key, val in zip(keys, vals)}
    return data

def main():
    # Load dataset
    particle_ids = ["1000022"] * 2
    dl = SLHALoader(
        particle_ids=particle_ids,
        feat_dir="./features",
        target_dir="./targets",
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
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


    # Load models

    layers = "[5, 20, 20, 1]"
    epochs = 2500
    leapfrogsteps = 512


    step = 2
    root_dir = "./models/"
    kernel = "hmc"
    pretraining_steps = [int(2 ** i) for i in range(5, 14, step)]
    model_fnames = [
        root_dir + f"kernel_{kernel}_results_1000_burnin_1000_epochs_{epoch}_leapfrogsteps_512_nodes_{layers}.npz"
        for epoch in pretraining_steps
    ]


    samplers = {
        "hmc": {
            "kernel_name": "HMC",
            "model_fnames": [
                                root_dir + f"kernel_hmc_results_1000_burnin_1000_epochs_{epoch}_leapfrogsteps_512_nodes_{layers}.npz"
                                for epoch in pretraining_steps
                            ],       
        },
        # "nuts": {
        #     "kernel_name": "NUTS",
        #     "model_fnames": [
        #         root_dir + f"kernel_nuts_results_1000_burnin_1000_epochs_{epoch}_leapfrogsteps_512_nodes_{layers}.npz"
        #         for epoch in pretraining_steps
        #     ],
        # },
    }

    # Load models
    for s in samplers:
        d = samplers.get(s)
        models = [BayesianNeuralNetwork() for _ in d.get("model_fnames")]
        for bnn, model_name in zip(models, d.get("model_fnames")):
            bnn.load_model(fname=model_name)
        d["models"] = models

    # Compute predictions
    for s in samplers:
        d = samplers.get(s)
        models = d.get("models")
        predictions = [bnn(x_test).numpy().squeeze(-1) for bnn in models]
        mean_predictions = [np.mean(y_pred, axis=0) for y_pred in predictions]
        std_predictions = [np.std(y_pred, axis=0) for y_pred in predictions]
        d["y_mean"] = mean_predictions
        d["y_std"] = std_predictions
    
    print(samplers)


    for s in samplers:
        d = samplers.get(s)
        y_mean = d.get("y_mean")
        r2_scores_log = [
            r2_score(y_true=y_test, y_pred=y_pred) for y_pred in y_mean
        ]
        predictions = [10 ** y for y in predictions]
        y_mean = [np.mean(y, axis=0) for y in predictions]
        r2_scores = [
            r2_score(y_true=10 ** y_test, y_pred=y_pred) for y_pred in y_mean
        ]


        plt.scatter(pretraining_steps, r2_scores, marker="x")
        plt.plot(pretraining_steps, r2_scores, label="Target space (HMC)")

        plt.scatter(pretraining_steps, r2_scores_log, marker="x")
        plt.plot(pretraining_steps, r2_scores_log, label="Log space (HMC)")

        plt.xlabel("Pretraining Epochs")
        plt.ylabel("$R^2$")
        plt.xscale("log", base=2)
        plt.legend()

        dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/r2_scores/"
        fname = "r2_score_vs_pretraining.pdf"
        plt.savefig(dir + fname)
    plt.close()



    # Standardized residuals in Log space
    

    y_preds = [bnn(x_test).numpy().squeeze(-1) for bnn in models]
    print([y.shape for y in y_preds])
    y_mean = [np.mean(y, axis=0) for y in y_preds]
    print([y.shape for y in y_mean])
    residuals = [y - y_test for y in y_mean]
    y_std = [np.std(y, axis=0) for y in y_preds]
    std_residuals = [(res / std).numpy() for res, std in zip(residuals, y_std)]
    print([p.shape for p in std_residuals])
    print(f"{y_test.shape = }")


    x_min, x_max = -5, 5
    for i, (residual, n) in enumerate(zip(std_residuals, pretraining_steps)):
        label = str(n)
        plt.hist(residual, histtype="step", density=True, bins=100, label=label)

    x = np.linspace(x_min, x_max, 1001)
    normal_dist = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    plt.plot(x, normal_dist, color="black", linestyle="--", label="$\mathcal{N}(0, 1)$")
    plt.xlim((x_min, x_max))
    plt.xlabel("Standardized Residual")
    plt.ylabel("Density")
    plt.legend()

    dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/standardized_residuals/effect_of_pretraining/"
    fname = f"standardized_residuals_hmc_vs_pretraining_steps.pdf"
    plt.savefig(dir + fname)



if __name__ == "__main__":
    main()
