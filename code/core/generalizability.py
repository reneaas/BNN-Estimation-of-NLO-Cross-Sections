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


def load_models():
    model_names = [
        f"models/{i}_hidden_layers_tanh.npz" for i in range(1, 6)
    ]

    models = [BayesianNeuralNetwork() for _ in model_names]

    for bnn, model_name in zip(models, model_names):
        bnn.load_model(fname=model_name)
    return models




def main():
    data = load_dataset(particle_ids=["1000022"] * 2)
    models = load_models()
    print(*models)


    x_train, y_train = data.get("train")
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = y_train.squeeze(-1)

    x_val, y_val = data.get("val")
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_val = y_val.squeeze(-1)

    x_test, y_test = data.get("test")
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = y_test.squeeze(-1)


    x_data = {
        "train": x_train,
        "val": x_val,
        "test": x_test,
    }

    y_data = {
        "train": y_train,
        "val": y_val,
        "test": y_test,
    }

    predictions = {}
    for key in x_data:
        predictions[key] = [
            np.mean(bnn(x_data.get(key)).numpy().squeeze(-1), axis=0) for bnn in models 
        ]

    
    r2_scores_log = {}
    for key in predictions:
        r2_scores_log[key] = [
            r2_score(y_true=y_data.get(key), y_pred=y_pred) for y_pred in predictions.get(key)
        ]
        print(len(r2_scores_log.get(key)))
    
    model_names = [str(i + 1) for i in range(len(models))]
    print(len(model_names))
    print(r2_scores_log)
    for key in r2_scores_log:
        print("got here")
        plt.scatter(model_names, r2_scores_log.get(key), marker="x")
        plt.plot(model_names, r2_scores_log.get(key), label=key)      

    plt.xlabel("Model Name")
    plt.ylabel("$R^2$")
    plt.legend()
    plt.show()


    for key in y_data:
        y_data[key] = 10 ** y_data.get(key)

    r2_scores = {}
    for key in x_data:
        predictions[key] = [
            np.mean(10 ** bnn(x_data.get(key)).numpy().squeeze(-1), axis=0) for bnn in models
        ]

    for key in predictions:
        r2_scores[key] = [
            r2_score(y_true=y_data.get(key), y_pred=y_pred) for y_pred in predictions.get(key)
        ]

    for key in r2_scores:
        print("got here")
        plt.scatter(model_names, r2_scores.get(key), marker="x")
        plt.plot(model_names, r2_scores.get(key), label=key)      

    plt.xlabel("Model Name")
    plt.ylabel("$R^2$")
    plt.legend()
    plt.show()

    


if __name__ == "__main__":
    main()

  





