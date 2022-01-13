import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import tqdm
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns


from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHAloader
from utils.preprocessing import split_data


def trace_fn_hmc(_, pkr):  
    return {
        # "target_log_prob": pkr.target_log_prob,
        # "leapfrogs_taken": pkr.leapfrogs_taken,
        # "has_divergence": pkr.has_divergence,
        # "energy": pkr.energy,
        # "log_accept_ratio": pkr.log_accept_ratio,
        "is_accepted": pkr.is_accepted
    }

def trace_fn_no_u_turn(_, pkr):
    return {
        "target_log_prob": pkr.target_log_prob,
        "leapfrogs_taken": pkr.leapfrogs_taken,
        "has_divergence": pkr.has_divergence,
        "energy": pkr.energy,
        "log_accept_ratio": pkr.log_accept_ratio,
        "is_accepted": pkr.is_accepted
    }


def trace_fn_adaptive_hmc(_, pkr):
    return {
        # "target_log_prob": pkr.inner_results.target_log_prob,
        # "leapfrogs_taken": pkr.inner_results.leapfrogs_taken,
        # "has_divergence": pkr.inner_results.has_divergence,
        # "energy": pkr.inner_results.energy,
        # "log_accept_ratio": pkr.inner_results.log_accept_ratio,
        "is_accepted": pkr.inner_results.is_accepted,
        # "step_size": pkr.inner_results.step_size,
        "target_accept_prob": pkr.target_accept_prob
    }

def trace_fn_adaptive_no_u_turn(_, pkr):
    return {
        "target_log_prob": pkr.inner_results.target_log_prob,
        "leapfrogs_taken": pkr.inner_results.leapfrogs_taken,
        "has_divergence": pkr.inner_results.has_divergence,
        "energy": pkr.inner_results.energy,
        "log_accept_ratio": pkr.inner_results.log_accept_ratio,
        "is_accepted": pkr.inner_results.is_accepted,
        "step_size": pkr.inner_results.step_size,
        "target_accept_prob": pkr.target_accept_prob
    }

def main():
    instruction = sys.argv[1]


    particle_ids = ["1000022", "1000022"]
    target_dir = "./targets"
    feat_dir = "./features"
    dl = SLHAloader(
        particle_ids=particle_ids,
        feat_dir=feat_dir,
        target_dir=target_dir,
        target_keys=["nlo"],
    )
    features = dl.features.to_numpy()
    targets = dl.targets.get("nlo").to_numpy()
    targets = np.log10(targets) #Transform targets to log10 space.
    nan_idx = np.isnan(targets)
    idx = (nan_idx == False)
    features = features[idx]
    targets = targets[idx]

    data = split_data(features=features, targets=targets)
    x_train, y_train = data["train"]



    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_train = y_train[:, None]
    input_size = x_train.shape[-1]
    layers = [input_size, 10, 20, 10, 1]
    activation = "relu"
    # activation = ["relu", "relu", "identity"]


    num_chains = 10
    num_results = 10
    num_burnin_steps = 1000
    if sys.argv[1] == "train":
        bnn = BayesianNeuralNetwork(
            layers=layers, activation=activation, num_chains=num_chains
        )

        start = time.perf_counter()
        loss = bnn.mle_fit(
            x=x_train,
            y=y_train,
            epochs=10_000,
            lr=0.001
        )
        end = time.perf_counter()
        timeused = end - start
        print(f"timeused = {timeused} on MLE fit.")

        plt.plot(loss)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()

        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            num_leapfrog_steps=100,
            step_size=[1e-3 for _ in range(len(bnn.weights))],
            target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train)
        )

        # kernel = tfp.mcmc.NoUTurnSampler(
        #     target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train),
        #     step_size=[1e-3 for _ in range(len(bnn.weights))]
        # )


        # kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        #     inner_kernel=kernel,
        #     num_adaptation_steps=int(0.8 * num_burnin_steps)
        # )

        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=int(0.8 * num_burnin_steps)
        )


        start = time.perf_counter()
        chain, trace = bnn.sample_chain(
            kernel=kernel,
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=0,
            fname="models/bnn_mssm_nouturn.npz",
            trace_fn=trace_fn_adaptive_hmc
        )
        end = time.perf_counter()
        timeused = end - start
        print(f"timeused = {timeused} on sampling")

        print(trace)
    
    elif sys.argv[1] == "test":
        x_test, y_test = data["test"]
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

        bnn = BayesianNeuralNetwork()
        bnn.load_model(fname="models/bnn_mssm_nouturn.npz")
        y_pred = bnn(x_test)

        y_pred = y_pred.numpy().squeeze(-1)
        y_mean = np.mean(y_pred, axis=0)

        rel_error = (y_test - y_mean) / y_test
        sns.histplot(rel_error, stat="density")
        plt.xlabel("Relative error")
        plt.title("Relative error log space")
        plt.figure()
        # plt.show()

        standardized_residual = (y_test - y_mean) / np.std(y_pred, axis=0)
        print(standardized_residual)
        sns.histplot(standardized_residual, stat="density")
        plt.xlabel("Standardized residual")
        # plt.hist(standardized_residual, bins=10)
        plt.title("Standardized residual log space")
        plt.figure()
        # plt.show()

        print(y_mean)
        print(f"{y_pred.shape=}")

        print(f"{y_test.shape=}")

        y_pred = 10 ** y_pred
        y_mean = np.mean(y_pred, axis=0)

        rel_error = (10 ** y_test - y_mean) / (10 ** y_test)
        print(rel_error.shape)
        sns.histplot(rel_error, stat="density")
        plt.xlabel("Relative error")
        plt.figure()
        # plt.show()

        standardized_residual = (10 ** y_test - y_mean) / np.std(y_pred, axis=0)
        sns.histplot(standardized_residual, stat="density")
        plt.xlabel("Standardized residual")
        plt.show()






if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()
