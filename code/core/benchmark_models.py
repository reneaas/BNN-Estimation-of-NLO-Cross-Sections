import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data


import sys


def main():

    #Load data
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

    #Load models
    model_names1 = [
        f"models/{i+1}_small_hidden_layers.npz" for i in range(4)
    ] 
    model_names2 = [
        f"models/{i+1}_hidden_layers_tanh.npz" for i in range(5)
    ]
    model_names = model_names1 + model_names2
    print(*model_names)

    models = [
        BayesianNeuralNetwork() for _ in model_names
    ]

    for bnn, model_name in zip(models, model_names):
        bnn.load_model(fname=model_name)
    print(*models)

    model_predictions = [
        bnn(x_test).numpy().squeeze(-1) for bnn in models
    ]

    model_predictions_mean = [
        np.mean(prediction, axis=0) for prediction in model_predictions
    ]
    print([p.shape for p in model_predictions_mean])

    model_predictions_std = [
        np.std(prediction, axis=0) for prediction in model_predictions
    ]

    rel_error = [
        (y_test - y_mean) / y_test for y_mean in model_predictions_mean
    ]
    rel_error = [err.numpy() for err in rel_error]

    std_residual = [
        (y_test - y_mean) / y_std for y_mean, y_std in zip(model_predictions_mean, model_predictions_std) 
    ]
    std_residual = [arr.numpy() for arr in std_residual]



    rel_error_mean = [
        np.mean(err) for err in rel_error
    ]

    rel_error_std = [
        np.std(err) for err in rel_error
    ]
    print(f"{rel_error_mean=}")
    print(f"{rel_error_std=}")

    num_params = [bnn.num_params for bnn in models]
    data = {
        "rel_error_mean": rel_error_mean,
        "rel_error_std": rel_error_std,
        "num_params": num_params,
    }
    df = pd.DataFrame(data)

    # plt.errorbar(x=df["num_params"], y=df["rel_error_mean"], yerr=df["rel_error_std"], capsize=5.0, fmt="o")
    # plt.xlabel("Number of parameters")
    # plt.ylabel("Relative error")
    # plt.show()

    rel_error = [err[err <= 1] for err in rel_error]
    rel_error = [err[-1 <= err] for err in rel_error]
    abs_rel_error = [abs(err) for err in rel_error]
    # sns.displot(rel_error, kind="ecdf", hue=[str(i) for i in num_params])
    # sns.ecdfplot(rel_error)
    # plt.show()

    print(len(rel_error))
    print(len(num_params))


    n_bins = 10
    fig = plt.hist(x=abs_rel_error, bins=n_bins, density=True, histtype="step", cumulative=True, label=num_params)
    plt.legend()
    plt.show()

    sys.exit()



    params = [
        [n] * err.shape[0] for n, err in zip(num_params, rel_error)
    ]
    print([err.shape for err in rel_error])
    rel_error = np.concatenate(rel_error, axis=0)
    params = np.concatenate(params, axis=0)
    std_residual = np.concatenate(std_residual, axis=0)
    data = {
        "Number of parameters": params,
        "Relative error": rel_error,
        "Standardized residual": std_residual,
    }
    df = pd.DataFrame(data)
    print(df)

    # sns.violinplot(x="Number of parameters", y="Relative error", data=df)
    # plt.show()

    # sns.violinplot(x="Number of parameters", y="Standardized residual", data=df)
    # plt.show()

    # sns.displot(data=df, x="Relative error", hue="Number of parameters", kind="ecdf")

    plt.hist(x=df["Relative error"])




if __name__ == "__main__":
    main()