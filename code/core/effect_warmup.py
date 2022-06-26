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

def load_models(model_fnames):
    models = [BayesianNeuralNetwork() for _ in model_fnames]
    for bnn, model_name in zip(models, model_fnames):
        bnn.load_model(fname=model_name)
    return models

def load_dataset(particle_ids):
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
    return data



def main():
    # Load models

    layers = "[5, 20, 20, 1]"
    epochs = 2500
    leapfrogsteps = 512


    step = 2
    root_dir = "./models/"
    kernel = "nuts"
    model_fnames = [
        root_dir + f"kernel_{kernel}_results_1000_burnin_{int(2 ** i)}_epochs_2500_leapfrogsteps_512_nodes_{layers}.npz"
        for i in range(5, 14, step)
    ]

    root_dir = "./results/"
    model_data_fnames = [
        root_dir + f"kernel_{kernel}_results_1000_burnin_{int(2 ** i)}_epochs_2500_leapfrogsteps_512_nodes_{layers}.txt"
        for i in range(5, 14, 1)
    ]

    model_data = [
        get_data(fname) for fname in model_data_fnames
    ]
    print(model_data)

    data_merged = {key: [] for key in model_data[0]}
    for d in model_data:
        for key in d:
            data_merged[key].append(d.get(key))
    print(data_merged)
    df = pd.DataFrame(data_merged)

    x = df["num_burnin_steps"].to_numpy(dtype=int)
    y = df["num_leapfrog_steps"].to_numpy(dtype=int)
    plt.plot(x, y)
    plt.scatter(x, y, label="datapoints", marker="^", color="red")
    plt.xscale("log", base=2)
    plt.xlabel("Number of Warm-up Steps")
    plt.ylabel("Avg. Leapfrog Steps")
    plt.legend()


    dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/standardized_residuals/effect_of_burnin/"
    fname = f"avg_burnin_steps_{kernel}_vs_burn_in_steps.pdf"
    # plt.savefig(dir + fname)
    plt.show()


    # Load dataset
    data = load_dataset(particle_ids=["1000022"] * 2)
    x_test, y_test = data.get("test")
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = y_test.squeeze(-1)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


    num_burnin_steps = [int(2 ** i) for i in range(5, 14, step)]
    
    models = load_models(model_fnames=model_fnames)

    model_data = {
        "models": models,
        "num_burnin_steps": num_burnin_steps,
        "model_fnames": model_fnames
    }
    

    y_preds = [bnn(x_test).numpy().squeeze(-1) for bnn in models]
    y_mean = [np.mean(y, axis=0) for y in y_preds]
    residuals = [y - y_test for y in y_mean]
    y_std = [np.std(y, axis=0) for y in y_preds]
    std_residuals = [(res / std).numpy() for res, std in zip(residuals, y_std)]


    x_min, x_max = -5, 5
    for i, (residual, n) in enumerate(zip(std_residuals, num_burnin_steps)):
        label = str(n)
        plt.hist(residual, histtype="step", density=True, bins=100, label=label)

    x = np.linspace(x_min, x_max, 1001)
    normal_dist = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    plt.plot(x, normal_dist, color="black", linestyle="--", label="$\mathcal{N}(0, 1)$")
    plt.xlim((x_min, x_max))
    plt.xlabel("Standardized Residual")
    plt.ylabel("Density")
    plt.legend()

    dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/standardized_residuals/effect_of_burnin/"
    fname = f"standardized_residuals_{kernel}_vs_burn_in_steps.pdf"
    # plt.savefig(dir + fname)
    plt.show()


    model_predictions = [bnn(x_test).numpy().squeeze(-1) for bnn in models]
    mean_predictions = [np.mean(p, axis=0) for p in model_predictions]
    predictions_std = [np.std(p, axis=0) for p in model_predictions]

    r2_scores_log = [
        r2_score(y_true=y_test, y_pred=y_pred) for y_pred in mean_predictions
    ]
    print(f"Log space: {r2_scores_log = }")

    # Target space calculations

    y_test = 10 ** y_test
    model_predictions = [10 ** y_pred for y_pred in model_predictions]
    mean_predictions = [np.mean(p, axis=0) for p in model_predictions]
    r2_scores = [
        r2_score(y_true=y_test, y_pred=y_pred) for y_pred in mean_predictions
    ]
    print(f"Target space: {r2_scores = }")

    with open(file=f"tab_data/r2_scores_burn_in_effect_{kernel}.txt", mode="w") as outfile:
        outfile.write(
            "warm_up_steps r2_log r2_target \n"
        )
        for b, r2_log, r2 in zip(num_burnin_steps, r2_scores_log, r2_scores):
            outfile.write(f"{b} {r2_log} {r2} \n")
    plt.plot(num_burnin_steps, r2_scores_log, color="black", label="Log space")
    plt.scatter(num_burnin_steps, r2_scores_log, color="red", marker="x")

    plt.plot(num_burnin_steps, r2_scores, color="blue", label="Target space")
    plt.scatter(num_burnin_steps, r2_scores, marker="x", color="red")
    plt.xlabel("Warm-up steps")
    plt.ylabel(r"$R^2$")
    plt.xscale("log", base=2)
    plt.legend()
    plt.show()



    # Compute r2 scores








if __name__ == "__main__":
    main()
