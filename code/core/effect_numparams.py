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
    targets = np.log10(targets[idx])
    targets = targets[:, None]
    features = features[idx]

    data = split_data(features=features, targets=targets)
    return data


def load_model_data():
    samplers = {
        "nuts": {
            "name": "NUTS",
            "model_fnames": [
                "kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 32, 1].npz",
                "kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 64, 1].npz",
                "kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 128, 1].npz",
                "kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 256, 1].npz",
                "kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 512, 1].npz",
                "kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 1024, 1].npz",
                "kernel_nuts_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 2048, 1].npz",
            ],
        },
        "hmc": {
            "name": "HMC",
            "model_fnames": [
                "kernel_hmc_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 32, 1].npz",
                "kernel_hmc_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 64, 1].npz",
                "kernel_hmc_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 128, 1].npz",
                "kernel_hmc_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 256, 1].npz",
                "kernel_hmc_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 512, 1].npz",
                "kernel_hmc_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 1024, 1].npz",
                "kernel_hmc_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 2048, 1].npz",
                "kernel_hmc_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 4096, 1].npz",
                "kernel_hmc_results_1000_burnin_1000_epochs_2500_leapfrogsteps_512_nodes_[5, 8192, 1].npz"
            ],
        },
    }

    for sampler in samplers:
        data = samplers.get(sampler)
        model_fnames = data.get("model_fnames")
        dir = "./models/"
        model_fnames = [dir + fname for fname in model_fnames]
        samplers[sampler]["model_fnames"] = model_fnames


    # Load model parameters
    for sampler in samplers:
        data = samplers.get(sampler)
        fnames = data.get("model_fnames")
        models = [BayesianNeuralNetwork() for _ in fnames]
        for bnn, fname in zip(models, fnames):
            bnn.load_model(fname=fname)

        samplers[sampler]["models"] = models
    

    # Compute number of parameters in model.

    for sampler in samplers:
        data = samplers.get(sampler)
        models = data.get("models")
        model_weights = [bnn.weights for bnn in models]
        shapes = [
            [w.numpy().shape[1:] for w in weights]
            for weights in model_weights
        ]
        # print(f"{shapes = }")
        hidden_layer_shapes = [shape[:2] for shape in shapes]
        hidden_layer_nodes = []
        for layer in hidden_layer_shapes:
            hidden_layer_nodes.append(layer[1][0])
        samplers[sampler]["hidden layer nodes"] = hidden_layer_nodes
        num_params = []
        for shape in shapes:
            p = 0
            for w_shape, b_shape in zip(shape[::2], shape[1::2]):
                p += math.prod(w_shape) + math.prod(b_shape)
            num_params.append(p)
        
        samplers[sampler]["num_params"] = num_params
    return samplers

def main():
    samplers = load_model_data()
    data = load_dataset(particle_ids=["1000022"] * 2)
    x_test, y_test = data.get("test")
    y_test = y_test.squeeze(-1)

    start = time.perf_counter()
    for sampler in samplers:
        models = samplers.get(sampler).get("models")
        predictions = []
        for model in models:
            tmp_y_pred = np.zeros(shape=(1000, x_test.shape[0])) 
            for i in trange(x_test.shape[0]):
                x = x_test[i, ...][None, :]
                x = tf.convert_to_tensor(x, dtype=tf.float32)
                y = model(x).numpy().squeeze(-1).squeeze(-1)
                tmp_y_pred[..., i] = y
            predictions.append(tmp_y_pred)
        samplers[sampler]["predictions log space"] = predictions


    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused = }")


    for sampler in samplers:
        sampler_data = samplers.get(sampler)
        sampler_name = sampler_data.get("name")
        predictions = sampler_data.get("predictions log space")
        hidden_layer_nodes = sampler_data.get("hidden layer nodes")
        num_params = sampler_data.get("num_params")


        # Compute r2 scores in log space
        y_mean = [np.mean(y, axis=0) for y in predictions]
        r2_scores_log = [
            r2_score(y_true=y_test, y_pred=y) for y in y_mean
        ]

        print(f"{r2_scores_log = }")

        predictions = [10 ** y for y in predictions]
        y_mean = [np.mean(y, axis=0) for y in predictions]
        r2_scores = [
            r2_score(y_true=10 ** y_test, y_pred=y) for y in y_mean
        ]

        print(f"{r2_scores = }")

        plt.scatter(hidden_layer_nodes, r2_scores_log, marker="x")
        plt.plot(hidden_layer_nodes, r2_scores_log, label=f"Log space ({sampler_name})")


        plt.scatter(hidden_layer_nodes, r2_scores, marker="x")
        plt.plot(hidden_layer_nodes, r2_scores, label=f"Target space ({sampler_name})")

    plt.xlabel("Number of Parameters")
    plt.ylabel("$R^2$")
    plt.xscale("log", base=2)
    plt.legend()
    plt.show()


        

        


        




if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()




