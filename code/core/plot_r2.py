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
    dir = "tab_data/"
    samplers = {
        "nuts": {
            "fname": dir + "r2_scores_burn_in_effect_nuts.txt",
        },
        "hmc": {
            "fname": dir + "r2_scores_burn_in_effect_hmc.txt"
        } 
    }
    for kernel in samplers:
        with open(file=samplers.get(kernel).get("fname"), mode="r") as infile:
            labels = infile.readline().split()
            print(labels)
            d = {key: [] for key in labels}
            for line in infile:
                vals = line.split()
                for label, val in zip(labels, vals):
                    d[label].append(float(val))
            d = {key: np.array(d.get(key)) for key in d}

        samplers[kernel]["data"] = d
    print(samplers)


    for kernel in samplers:
        kernel_name = "HMC" if kernel == "hmc" else "NUTS"
        d = samplers.get(kernel).get("data")
        print(d)
        idx = d.get("r2_log") >= 0
        plt.scatter(d.get("warm_up_steps")[idx], d.get("r2_log")[idx], marker="x")
        plt.plot(d.get("warm_up_steps")[idx], d.get("r2_log")[idx], label=f"Log space ({kernel_name})")

        idx = d.get("r2_target") >= 0
        plt.scatter(d.get("warm_up_steps")[idx], d.get("r2_target")[idx], marker="x")
        plt.plot(d.get("warm_up_steps")[idx], d.get("r2_target")[idx], label=f"Target space ({kernel_name})")

    plt.xlabel("Warm-up Steps")
    plt.ylabel(r"$R^2$")
    plt.xscale("log", base=2)
    plt.legend()

    dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/r2_scores/"
    fname = dir + "effect_of_burnin_r2_scores.pdf"
    plt.savefig(fname)
    # plt.show()

if __name__ == "__main__":
    main()
