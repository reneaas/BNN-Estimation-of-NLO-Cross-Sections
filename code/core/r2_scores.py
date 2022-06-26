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
from utils.metrics import r2_score



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


    model_names = [
        f"models/{i}_hidden_layers_tanh.npz" for i in range(1, 6)
    ]

    # layers = "[5, 20, 20, 1]"
    # epochs = 2500
    # leapfrogsteps = 512
    # step = 2
    # root_dir = "./models/"
    # kernel = "hmc"
    # model_names = [
    #     root_dir + f"kernel_{kernel}_results_1000_burnin_{int(2 ** i)}_epochs_2500_leapfrogsteps_512_nodes_{layers}.npz"
    #     for i in range(5, 14, step)
    # ]

    models = [BayesianNeuralNetwork() for _ in model_names]

    for bnn, model_name in zip(models, model_names):
        bnn.load_model(fname=model_name)
    print(*models)
    print([[w.shape for w in bnn.weights] for bnn in models])

    # Log space calculations
    data = split_data(features=features, targets=targets)
    x_test, y_test = data.get("test")
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = y_test.squeeze(-1)


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

    with open(file="tab_data/r2_scores_tabulated_models.txt", mode="w") as outfile:
        outfile.write(
            "model r2_log r2_target \n"
        )
        for i, (r2_log, r2) in enumerate(zip(r2_scores_log, r2_scores)):
            outfile.write(f"{{i + 1}} {{r2_log}} {{r2}} \n")
    






 
if __name__ == "__main__":
    main()