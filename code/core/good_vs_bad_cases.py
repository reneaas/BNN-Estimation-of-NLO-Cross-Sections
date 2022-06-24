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



def main():
    model_fname = "models/3_hidden_layers_tanh.npz"
    bnn = BayesianNeuralNetwork()
    bnn.load_model(fname=model_fname)
    print(bnn)


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



   



if __name__ == "__main__":
    main()