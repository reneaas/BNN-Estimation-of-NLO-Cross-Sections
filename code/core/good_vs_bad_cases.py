import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_probability as tfp
import pandas as pd
import time
from tqdm import trange

from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data
import sys 


def load_models():
    model_fname = "models/3_hidden_layers_tanh.npz"
    bnn = BayesianNeuralNetwork()
    bnn.load_model(fname=model_fname)
    print(bnn)
    return bnn

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

def good_vs_bad_cases():
    bnn = load_models()
    data = load_dataset(particle_ids=["1000022"] * 2)
    x_test, y_test = data.get("test")
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = y_test.squeeze(-1)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)


    num_trials = 50
    dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/predictive_distributions/"
    for _ in trange(num_trials):
        idx = np.random.randint(0, x_test.shape[0])
        print(f"{idx = }")
        x = x_test[idx][None, :]
        y = y_test[idx]
        y_pred = bnn(x)

        y_pred = y_pred.numpy().squeeze(-1).squeeze(-1)
        y_mean = np.mean(y_pred, axis=0)
        plt.hist(y_pred, histtype="step", density=True, bins=100)
        plt.axvline(y_mean, color="black", label="Sample mean")
        plt.axvline(y, color="red", label="True target")
        plt.xlabel("Predicted Value")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(dir + f"predictive_distribution_point_idx_{idx}.pdf")
        plt.close()

def confidence_interval(data_type="test"):
    bnn = load_models()
    data = load_dataset(particle_ids=["1000022"] * 2)
    x_test, y_test = data.get(data_type)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = y_test.squeeze(-1)
    # y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    predictions = bnn(x_test).numpy().squeeze(-1)

    sample_mean = np.mean(predictions, axis=0)
    sample_std = np.std(predictions, axis=0)

    idx_above = (sample_mean - sample_std <= y_test)
    idx_below = (y_test <= sample_mean + sample_std)
    idx = (idx_above == idx_below)
    cumulative_confidence = []
    stddevs = [k for k in range(1, 6)]
    # stddevs = np.linspace(0.1, 5, 15)
    for i, k in enumerate(stddevs):
        # stddevs.append(k)
        idx_above = 1. * (sample_mean - k * sample_std <= y_test)
        idx_below = 1. * (y_test <= sample_mean + k * sample_std)
        n = np.sum(idx_above * idx_below)
        print(f"{n = } ; {n / y_test.shape[0] = } ; {k = }")
        cumulative_confidence.append(n / y_test.shape[0])


    return stddevs, cumulative_confidence



def sample_std():
    bnn = load_models()
    data = load_dataset(particle_ids=["1000022"] * 2)
    x_test, y_test = data.get("test")
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = y_test.squeeze(-1)

    predictions = bnn(x_test).numpy().squeeze(-1)
    sample_mean = np.mean(predictions, axis=0)
    sample_std = np.std(predictions, axis=0)

    sample_std_target = [std * 10 ** mean * np.log(10) for std, mean in zip(sample_std, sample_mean)]
    print(f"{sample_std = }")
    print(f"{sample_std_target = }")
    predictions = [10 ** y for y in predictions]
    sample_std_target = [np.std(y, axis=0) for y in predictions]
    print(f"{sample_std_target = }")


    



   



if __name__ == "__main__":
    sample_std()
    # data_types = ["train", "val", "test"]
    # for d in data_types:
    #     stddevs, cumulative_confidence = confidence_interval(data_type=d)
    #     plt.plot(stddevs, cumulative_confidence, label=d)
    #     plt.scatter(stddevs, cumulative_confidence, marker="x")
    #     plt.axhline(y=0.95, linestyle="--", color="black")

    # plt.xlabel("$k$")
    # plt.ylabel("Percentage of Predictions")
    # plt.legend()
    # dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/confidence_estimation/"
    # fname = dir + "good_vs_bad_cases_confidence.pdf"
    # plt.savefig(fname)
    # plt.close()