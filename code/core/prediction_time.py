import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_probability as tfp
import pandas as pd
import time

from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data



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

    data = split_data(features=features, targets=targets)
    x_test, y_test = data.get("test")
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = y_test.squeeze(-1)


    model_names = [
        f"models/{i}_hidden_layers_tanh.npz" for i in range(1, 6)
    ]
    models = [BayesianNeuralNetwork() for _ in model_names]

    for bnn, model_name in zip(models, model_names):
        bnn.load_model(fname=model_name)
    print(*models)
    print([[w.shape for w in bnn.weights] for bnn in models])


    num_trials = 10_000
    timeused = np.zeros(shape=(num_trials, len(models)))
    for i in range(num_trials):
        for j, bnn in enumerate(models):
            x = np.random.normal(size=(1, 5))
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            start = time.perf_counter()
            y = bnn(x)
            y_mean = tf.reduce_mean(y, axis=0)
            y_std = tf.math.reduce_std(y, axis=0)
            end = time.perf_counter()
            timeused[i, j] = end - start

    timeused = np.mean(timeused, axis=0)
    print(f"{timeused = }")

    model_names = [str(i + 1) for i in range(len(models))]
    timeused = np.array(timeused)
    plt.scatter(model_names, timeused * 1e3, color="red", label="Datapoints")
    plt.plot(model_names, timeused * 1e3)
    plt.xlabel("Model")
    plt.ylabel("Time used [ms]")
    plt.legend()

    path = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/prediction_time/"
    fname = "prediction_time.pdf"
    plt.savefig(path + fname)

if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()


