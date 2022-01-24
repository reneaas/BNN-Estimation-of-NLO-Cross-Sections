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


from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHAloader
from utils.preprocessing import split_data
from utils.trace_functions import (
    trace_fn_hmc,
    trace_fn_no_u_turn,
    trace_fn_adaptive_hmc,
    trace_fn_adaptive_no_u_turn,
)


def main(
    particle_ids,
    fname,
    num_chains,
    num_results,
    num_burnin_steps,
    num_epochs,
    batch_size,
    instruction,
    kernel,
    trace,
    num_steps_between_results,
):
        
    # particle_ids = ["1000022", "1000022"]
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
    targets = np.log10(targets)  # Transform targets to log10 space.
    nan_idx = np.isnan(targets)
    idx = nan_idx == False
    features = features[idx]
    targets = targets[idx]

    data = split_data(features=features, targets=targets)
    x_train, y_train = data["train"]

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_train = y_train[:, None]
    input_size = x_train.shape[-1]
    layers = [input_size, 50, 50, 50, 50, 1]
    activation = "tanh"
    # activation = ["relu", "relu", "identity"]

    if instruction == "train":
        bnn = BayesianNeuralNetwork(
            layers=layers, activation=activation, num_chains=num_chains, lamb=1e-3
        )

        start = time.perf_counter()
        loss = bnn.mle_fit(
            x=x_train,
            y=y_train,
            epochs=num_epochs,
            lr=0.001,
            batch_size=batch_size,
        )
        end = time.perf_counter()
        timeused = end - start
        print(f"timeused = {timeused} on MLE fit.")

        # plt.plot(loss)
        # plt.xlabel("epochs")
        # plt.ylabel("loss")
        # plt.show()

        if trace is False:
            trace_fn = None
        else:
            if kernel == "hmc":
                trace_fn = lambda _, pkr: trace_fn_adaptive_hmc(_, pkr)
            elif kernel == "no_u_turn":
                trace_fn = lambda _, pkr: trace_fn_adaptive_no_u_turn(_, pkr)

        if kernel == "hmc":
            kernel = tfp.mcmc.HamiltonianMonteCarlo(
                num_leapfrog_steps=300,
                step_size=[tf.fill(w.shape, 0.001) for w in bnn.weights],
                target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train),
            )
        elif kernel == "no_u_turn":
            kernel = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train),
                step_size=[tf.fill(w.shape, 0.001) for w in bnn.weights],
            )

        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel, 
            num_adaptation_steps=int(0.8 * num_burnin_steps),
            target_accept_prob=0.75,
        )


        start = time.perf_counter()
        if trace_fn is None:
            chain = bnn.sample_chain(
                kernel=kernel,
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                num_steps_between_results=num_steps_between_results,
                fname=fname,
                trace_fn=trace_fn,
            )
        else:
            chain, trace = bnn.sample_chain(
                kernel=kernel,
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                num_steps_between_results=0,
                fname=fname,
                trace_fn=trace_fn,
            )
            print(trace)
            accept_ratio = sum(trace.get("is_accepted").numpy()) / num_results
            print(f"{accept_ratio=}")

        end = time.perf_counter()
        timeused = end - start
        print(f"timeused = {timeused} on sampling")

    elif instruction == "test":
        x_test, y_test = data["test"]
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

        bnn = BayesianNeuralNetwork()
        bnn.load_model(
            fname=fname,
        )
        y_pred = bnn(x_test)

        y_pred = y_pred.numpy().squeeze(-1)
        y_mean = np.mean(y_pred, axis=0)

        rel_error = (y_test - y_mean) / y_test
        print(f"{tf.reduce_mean(rel_error)=}")
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--process", help="""Provide particle process. e.g (1000022, 1000022)""",
    )
    parser.add_argument("--train", help="Train the model", action="store_true")
    parser.add_argument("--test", help="Test the model", action="store_true")
    parser.add_argument("-f", "--file", help="Filename to save trained model.")
    parser.add_argument("--epochs", help="Number of epochs to pretrain model", type=int)
    parser.add_argument("--batch", help="Batch size to train the model with", type=int)
    parser.add_argument(
        "--chains", help="number of MCMC chains to run in parallel (SIMD fashion)", type=int
    )
    parser.add_argument("--results", help="Number of results in the MCMC chains", type=int)
    parser.add_argument("--burn", help="Number of burn-in iterations", type=int)
    parser.add_argument("--cpu", help="Run on CPU", action="store_true")
    parser.add_argument("--gpu", help="Run on GPU", action="store_true")
    parser.add_argument(
        "--kernel", help="Kernel to run MCMC chain. Options: {`hmc`, `no_u_turn`}."
    )
    parser.add_argument(
        "--trace",
        help="Boolean specifying to gather trace statistics.",
        action="store_true",
    )
    parser.add_argument("--skip", help="Number of steps between results. Thinning", type=int)
    args = parser.parse_args()

    if args.cpu:
        device = "/CPU:0"
    elif args.gpu:
        device = "/GPU:0"
    else:
        device = "/CPU:0"

    if args.process is None:
        raise ValueError("Process not specified.")
    try:
        process = list(eval(args.process))
        process = [str(id) for id in process]
    except TypeError:
        print(
            f"process = {args.process} is not specified with the correct format. Specify as a list or tuple."
        )

    if args.train is True:
        instruction = "train"
    elif args.test is True:
        instruction = "test"
    else:
        instruction = "train"

    print("Process: ", process)
    print("Burn in samples:", args.burn)
    print("Number of results", args.results)
    print("Number of steps between", args.skip)
    print("Kernel:", args.kernel)


    with tf.device(device):
        main(
            particle_ids=process,
            fname=args.file,
            num_chains=args.chains if args.chains is not None else 10,
            num_results=args.results,
            num_burnin_steps=args.burn,
            num_epochs=args.epochs,
            batch_size=args.batch if args.batch is not None else None,
            instruction=instruction,
            kernel=args.kernel,
            trace=args.trace,
            num_steps_between_results=args.skip if args.skip is not None else 0
        )

    # with tf.device("/CPU:0"):
    #     main()
