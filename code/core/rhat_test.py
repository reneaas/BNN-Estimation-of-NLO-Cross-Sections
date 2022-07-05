import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import tqdm
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import math
plt.rc("text", usetex=True)


from bnn.bnn import BayesianNeuralNetwork
from bnn.bnn_base import _BNNBase
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data
from utils.metrics import r2_score
from utils.trace_functions import (
    trace_fn_hmc,
    trace_fn_nuts,
    trace_fn_adaptive_hmc,
    trace_fn_adaptive_nuts,
)

num_chains = 10


fname = "./models/multi_chain_model_hmc4.npz"

bnn = BayesianNeuralNetwork()
bnn.load_model(fname=fname)

weights = bnn.weights
print(f"{[w.shape for w in bnn.weights] = }")
rhat = tfp.mcmc.potential_scale_reduction(weights)
rhat = [r.numpy().ravel() for r in rhat]
rhat = np.concatenate(rhat, axis=0)

density = True
cumulative = False
plt.hist(rhat, bins=100, histtype="step", cumulative=cumulative, density=density)
plt.xscale("log", base=10)
plt.xlabel(r"$\hat{R}$")
plt.ylabel("Density")
plt.axvline(1.1, linestyle="--", color="red")
plt.show()
# print(f"{rhat = }")

# Plot histograms over the weights from two different distributions
# X = weights[0][:, 0, 1, 2].numpy()
# Y = weights[0][:, 2, 1, 2].numpy()
# print(f"{X.shape = }")
# print(f"{Y.shape = }")
# for i in range(num_chains):
#     sns.kdeplot(weights[0][:, i, 1, 2].numpy())
# sns.kdeplot(X, color="blue")

# plt.hist(X, histtype="step", bins=100)
# plt.figure()
# plt.hist(Y, histtype="step", bins=100)
# sns.kdeplot(Y, color="red")
# sns.jointplot(data=df, x=xlabel, y=ylabel, kind="kde")
# plt.show()


particle_ids = ["1000022", "1000022"]
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
x_train, y_train = data["train"]

x_test, y_test = data.get("train")
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

y_pred = np.zeros(shape=(4096, num_chains, y_test.shape[0]))
with tf.device("/CPU:0"):
    # for i in tqdm.trange(x_test.shape[0]):
    #     x = x_test[i]
    #     x = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)
    #     y = bnn(x).numpy().squeeze(-1)
    #     y_pred[..., i] = y.squeeze(-1)
    y_pred = bnn(x_test).numpy().squeeze(-1)
print(f"{y_pred.shape = }")
y_pred = y_pred[:, ...]
y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
print(f"Computing rhat....")
rhat = tfp.mcmc.potential_scale_reduction(y_pred)
rhat = rhat.numpy()
print(np.where(rhat < 1.1))
print(f"{rhat.shape = }")
print(f"y {rhat = }")
plt.hist(rhat, histtype="step", bins=250, cumulative=False, density=density)
plt.axvline(1.1, linestyle="--", color="red")
plt.xlabel(r"$\hat{R}$")
plt.ylabel("Density")
plt.show()


# Plot predictive distribution of a specific input for all chains


with tf.device("/CPU:0"):
    y_pred = bnn(x_test).numpy().squeeze(-1)

print(f"{y_pred.shape = }")

for i in range(num_chains):
    sns.kdeplot(y_pred[:, i, 21])
    # plt.axvline(np.mean(y_pred[:, i, 21], axis=0))
plt.show()
