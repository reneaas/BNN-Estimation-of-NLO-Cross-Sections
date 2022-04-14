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
from bnn.bnn_base import _BNNBase
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data
from utils.trace_functions import (
    trace_fn_hmc,
    trace_fn_nuts,
    trace_fn_adaptive_hmc,
    trace_fn_adaptive_nuts,
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
    layers,
    activations,
):

    # particle_ids = ["1000022", "1000022"]
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

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    input_size = x_train.shape[-1]
    # layers = [input_size, 10, 10, 10, 10, 1]
    # activations = "swish"

    if instruction == "train":
        bnn = BayesianNeuralNetwork(
            layers=layers,
            activations=activations,
            num_chains=num_chains,
            lamb=1e-3,
            likelihood_noise=1.,
        )

        start = time.perf_counter()
        loss = bnn.mle_fit(
            x_train=x_train,
            y_train=y_train,
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
            elif kernel == "nuts":
                trace_fn = lambda _, pkr: trace_fn_adaptive_nuts(_, pkr)

        if kernel == "hmc":
            inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                num_leapfrog_steps=300,
                step_size=[tf.fill(w.shape, 0.001) for w in bnn.weights],
                target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train),
            )
        elif kernel == "nuts":
            inner_kernel = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train),
                step_size=[0.0001 for w in bnn.weights],
                max_tree_depth=12,
            )

        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner_kernel,
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
                num_steps_between_results=num_steps_between_results,
                fname=fname,
                trace_fn=trace_fn,
            )
            accept_ratio = sum(trace.get("is_accepted").numpy()) / num_results
            print(f"accept ratio = {accept_ratio}")
            if isinstance(inner_kernel, tfp.mcmc.NoUTurnSampler):
                print("Mean number of leapfrog steps taken: ", tf.reduce_mean(trace["leapfrogs_taken"]).numpy())

        end = time.perf_counter()
        timeused = end - start
        print(f"timeused = {timeused} on sampling")

    elif instruction == "test":
        x_test, y_test = data["test"]
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        y_test = y_test.squeeze(-1)

        bnn = BayesianNeuralNetwork()
        bnn.load_model(
            fname=fname,
        )
        print(bnn)

        start = time.perf_counter()
        y_pred = bnn(x_test)
        end = time.perf_counter()
        timeused = end - start
        print("timeused = ", timeused)

        # test_point = x_test[0, ...][None, :]
        # true_point = y_test[0, ...]
        # print(f"{test_point.shape=}")
        # predicted_point = bnn(test_point).numpy().squeeze(-1)
        # sns.histplot(predicted_point.ravel())
        # plt.axvline(true_point, label="True value", color="red")
        # plt.legend()
        # plt.show()



        y_pred = y_pred.numpy().squeeze(-1)
        y_mean = np.mean(y_pred, axis=0)

        rel_error = (y_test - y_mean) / y_test
        print(f"{tf.reduce_mean(rel_error)=}")
        sns.histplot(rel_error.ravel(), stat="density")
        plt.xlabel("Relative error")
        plt.title("Relative error log space")
        plt.figure()
        # plt.show()

        # x = np.linspace(-2, 2, 1000)
        standardized_residual = (y_test - y_mean) / np.std(y_pred, axis=0)
        print(standardized_residual.shape)
        # plt.plot(x, np.exp(-0.5 * x ** 2))
        sns.histplot(standardized_residual.ravel(), stat="density")
        # plt.hist(standardized_residual)
        plt.xlabel("Standardized residual")
        # plt.hist(standardized_residual, bins=10)
        plt.title("Standardized residual log space")
        plt.figure()

        print(y_mean)
        print(f"{y_pred.shape=}")

        print(f"{y_test.shape=}")

        # y_test = list(y_test) * y_pred.shape[0]
        # y_test = np.array(y_test)
        # y_pred = y_pred.ravel()
        # df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
        # sns.lmplot(x="y_test", y="y_pred", data=df)
        # plt.show()

        # for i in range(y_pred.shape[0]):
        #     plt.scatter(y_test, y_pred[i])
        # plt.show()

        y_pred = 10 ** y_pred
        y_mean = np.mean(y_pred, axis=0)


        rel_error = (10 ** y_test - y_mean) / (10 ** y_test)
        print(f"{rel_error.shape=}")
        sns.histplot(rel_error.ravel())
        # plt.hist(rel_error.ravel())
        plt.xlabel("Relative error")
        plt.figure()
        # plt.show()



        standardized_residual = (10 ** y_test - y_mean) / np.std(y_pred, axis=0)
        # plt.hist(standardized_residual.ravel())
        sns.histplot(standardized_residual.ravel())
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
        "--kernel", help="Kernel to run MCMC chain. Options: {`hmc`, `nuts`}."
    )
    parser.add_argument(
        "--trace",
        help="Boolean specifying to gather trace statistics.",
        action="store_true",
    )
    parser.add_argument("--skip", help="Number of steps between results. Thinning", type=int)
    parser.add_argument("--arch", help="The architecture of the model: [input_size, layer:0, layer:1, ..., layer:L, output_size]")
    parser.add_argument("--act", help="Activation function in hidden layers", type=str)
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
    
    if args.arch is None:
        raise ValueError("Architecture not specified.")
    try:
        print(args.arch)
        layers = list(eval(args.arch))
    except TypeError:
        print(f"layers = {args.arch} are not the correct type. Should be a list.")

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
            num_steps_between_results=args.skip if args.skip is not None else 0,
            layers=layers,
            activations="swish" if args.act is None else args.act
        )

    # with tf.device("/CPU:0"):
    #     main()
