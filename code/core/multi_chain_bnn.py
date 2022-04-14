import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import time
import argparse
import os

from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data
from utils.trace_functions import trace_fn_adaptive_hmc, trace_fn_adaptive_nuts


def main(config):
    separator = "_"
    path = "models/"
    fname = path + "multi_chain_model.npz"
    target_dir = "./targets"
    feat_dir = "./features"
    dl = SLHALoader(
        particle_ids=config.get("particle_ids"),
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
    x_train, y_train = data.get("train")

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    bnn = BayesianNeuralNetwork(
        layers=layers,
        activations=config.get("activations"),
        num_chains=config.get("num_chains"),
        lamb=1e-3,
        likelihood_noise=1.,
    )

    start = time.perf_counter()
    bnn.mle_fit(
        x_train=x_train,
        y_train=y_train,
        epochs=config.get("num_epochs"),
        lr=0.001,
        batch_size=config.get("batch_size"),
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"timeused = {timeused} on MLE fit.")

    if config.get("kernel") == "hmc":
        inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            num_leapfrog_steps=config.get("num_leapfrog_steps"),
            step_size=[tf.fill(w.shape, 0.001) for w in bnn.weights],
            target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train),
        )
        trace_fn = lambda _, pkr: trace_fn_adaptive_hmc(_, pkr)
    elif config.get("kernel") == "nuts":
        inner_kernel = tfp.mcmc.NoUTurnSampler(
            step_size=[tf.fill(w.shape, 0.001) for w in bnn.weights],
            target_log_prob_fn=bnn.get_target_log_prob_fn(x=x_train, y=y_train),
            max_tree_depth=10,
        )
        trace_fn = lambda _, pkr: trace_fn_adaptive_nuts(_, pkr)
    


    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=inner_kernel,
        num_adaptation_steps=config.get("num_burnin_steps"),
        target_accept_prob=0.75,
    )

    start = time.perf_counter()
    results = bnn.sample_chain(
        kernel=kernel,
        num_results=config.get("num_results"),
        num_burnin_steps=config.get("num_burnin_steps"),
        num_steps_between_results=config.get("num_steps_between_results"),
        fname=fname,
        trace_fn=trace_fn,
    )
    end = time.perf_counter()
    timeused = end - start
    chain, trace = results
    

    accept_ratio = sum(trace.get("is_accepted").numpy()) / config.get("num_results")
    print(f"accept ratio = {accept_ratio}")
    config["accept_ratio"] = accept_ratio


    print(f"timeused = {timeused} on sampling")
    config["timeused"] = timeused
    if isinstance(inner_kernel, tfp.mcmc.NoUTurnSampler):
        config["num_leapfrog_steps"] = tf.reduce_mean(trace["leapfrogs_taken"]).numpy()

    path = "./results/"
    if not os.path.exists(path):
        os.makedirs(path)
    fname = fname.replace(".npz", ".txt")
    fname = fname.strip("models/")
    fname = path + fname
    with open(fname, "w") as outfile:
        for key in config:
            outfile.write(f"{key}" + " ")
        outfile.write("\n")
        for key in config:
            outfile.write(f"{config.get(key)}" + " ")
        outfile.write("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--process", help="""Provide particle process. e.g (1000022, 1000022)""",
    )
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
    parser.add_argument("--skip", help="Number of steps between results. Thinning", type=int)
    parser.add_argument("--arch", help="The architecture of the model: [input_size, layer:0, layer:1, ..., layer:L, output_size]")
    parser.add_argument("--act", help="Activation function in hidden layers", type=str)
    parser.add_argument("--leapfrog", help="Number of leapfrog steps.", type=int)
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

    print("Process: ", process)
    print("Burn in samples:", args.burn)
    print("Number of results", args.results)
    print("Number of steps between", args.skip)
    print("Kernel:", args.kernel)

    config = {
        "particle_ids": process, 
        "kernel": args.kernel,
        "num_chains": args.chains,
        "num_results": args.results,
        "num_burnin_steps": args.burn,
        "num_epochs": args.epochs,
        "batch_size": args.batch if args.batch is not None else None,
        "num_steps_between_results": args.skip if args.skip is not None else 0,
        "layers": layers,
        "activations": args.act if args.act is not None else "tanh",
        "num_leapfrog_steps": args.leapfrog,
    }


    with tf.device(device):
        main(config)
