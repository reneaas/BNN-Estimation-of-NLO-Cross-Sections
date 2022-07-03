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
    # targets = np.log10(targets[idx])
    targets = targets[idx]
    targets = targets[:, None]
    features = features[idx]

    data = split_data(features=features, targets=targets)
    return data


def main():
    data = load_dataset(particle_ids=["1000022"] * 2)
    x_train, y_train = data.get("train")
    x_val, y_val = data.get("val")
    x_test, y_test = data.get("test")

    x = np.concatenate((x_train, x_val, x_test), axis=0)
    y = np.concatenate((y_train, y_val, y_test), axis=0)
    print(f"{x.shape = }")
    print(f"{y.shape = }")

    xlabels = [
        r"$m_{\tilde{\chi}_1^0}$",
        r"$N_{11}$",
        r"$N_{12}$",
        r"$N_{13}$",
        r"$N_{14}$",
    ]

    y_train = y_train.squeeze(-1)
    for i, xlabel in enumerate(xlabels[1:]):
        plt.subplot(2, 2, i + 1)
        plt.scatter(x_train[..., i + 1], y_train)
        plt.xlabel(xlabel)
        plt.ylabel(r"$\sigma_{\tilde{\chi}_1^0 \tilde{\chi}_1^0}$")
        plt.yscale("log", base=10)
    # plt.ylabel(r"$\sigma_{\tilde{\chi}_1^0 \tilde{\chi}_1^0}$")
    
    plt.show()


    plt.scatter(x_train[..., 0], y_train)
    plt.xlabel(xlabels[0])
    plt.ylabel(r"$\sigma_{\tilde{\chi}_1^0 \tilde{\chi}_1^0}$")
    plt.yscale("log", base=10)
    
    dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/dataset/"
    fname = dir + "masses.pdf"
    # plt.savefig(fname)
    plt.show()



if __name__ == "__main__":
    main()



