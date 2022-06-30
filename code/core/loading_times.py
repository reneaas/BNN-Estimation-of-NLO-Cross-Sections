import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rc("text", usetex=True)
import time 
import tensorflow as tf
from tqdm import trange

from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data


def load_model(fname):
    model = BayesianNeuralNetwork()
    model.load_model(fname=fname)
    return model


def main():


    time_measurements = []
    time_std = []
    model_names = [
        f"models/{i}_hidden_layers_tanh.npz" for i in range(1, 6)
    ]
    num_trials = 1000
    for name in model_names:
        tmp = []
        for _ in trange(num_trials):
            start = time.perf_counter()
            model = load_model(fname=name)
            end = time.perf_counter()
            tmp.append(end - start)
            del model
        time_measurements.append(tmp)

    for i, (name, t) in enumerate(zip(model_names, time_measurements)):
        plt.hist(t, histtype="step", bins=100, label=f"Model {i+1}", density=True)
    plt.xscale("log", base=10)
    plt.xlabel("Loading Times [s]")
    plt.ylabel("Density")
    plt.legend()

    dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/computational_cost/"
    fname = "loading_times.pdf"
    plt.savefig(dir + fname)


    plt.close()
    plt.hist(time_measurements[0], histtype="step", bins=100)
    plt.show()
    





if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()