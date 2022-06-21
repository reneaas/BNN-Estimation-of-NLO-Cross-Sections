import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

std_config = {
    "num_burnin_steps": 1000,
    "batch_size": 32,
    "num_results": 1000,
    "num_steps_between_results": 10,
    "num_burnin_steps": 1000,
    "num_epochs": 2500,
}

num_samples = (
    std_config.get("num_results") * std_config.get("num_steps_between_results") 
    + std_config.get("num_burnin_steps")
)

def plot_data(x, y, xlabel, ylabel, log_scale=True, xbase=None, ybase=None, fname=None):
    if not isinstance(x, np.ndarray):
        x = x.to_numpy()
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()
    idx = np.argsort(x, axis=0)
    x = x[idx]
    y = y[idx]
    plt.plot(x, y)
    plt.scatter(x, y, label="datapoints", color="red", marker="^")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_scale:
        if xbase is not None:
            plt.xscale("log", base=xbase)
        if ybase is not None:
            plt.yscale("log", base=ybase)
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()

def load_results():
    fname = "./results/results_merged.pkl"
    df = pd.read_pickle(fname)
    return df


def time_vs_leapfrogsteps_hmc():
    df = load_results()
    kernel = "hmc"
    num_params = 561
    df = df[df["kernel"] == kernel]
    df = df[df["num_params"] == num_params]
    df = df[df["num_epochs"] == 2500]
    df = df[df["num_burnin_steps"] == 1000]
    x = df["num_leapfrog_steps"]
    y = df["timeused"] / num_samples
    # y = y.to_numpy()
    # y /= y[0]
    dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/computational_cost/"
    fname = "time_vs_leapfrogsteps_hmc.pdf"
    plot_data(
        x=x, 
        y=y, 
        xlabel="Number of Leapfrog steps",
        ylabel="Time per Sample [s]",
        xbase=2,
        ybase=2,
        fname=dir + fname,
    )

def time_vs_parameters_hmc():
    df = load_results()
    for key in std_config:
        df = df[df[key] == std_config.get(key)]
    df = df[df["kernel"] == "hmc"]
    df = df[df["num_leapfrog_steps"] == 512]

    x = df["num_params"]
    y = df["timeused"] / num_samples

    dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/computational_cost/"
    fname = "time_vs_params.pdf"
    fig = plot_data(
        x=x,
        y=y,
        xlabel="Number of Parameters",
        ylabel="Time Per Sample [s]",
        log_scale=True,
        xbase=10,
        ybase=10,
        fname=dir + fname,
    )


def leapfrog_steps_vs_parameters_nuts():
    df = load_results()
    df = df[df["kernel"] == "nuts"]
    for key in std_config:
        df = df[df[key] == std_config.get(key)]
    x = df["num_params"]
    y = df["num_leapfrog_steps"]
    y /= y[0]
    plot_data(
        x=x,
        y=y,
        xlabel="Number of Parameters",
        ylabel="Avg. Number of Leapfrog Steps",
        log_scale=True,
        xbase=2,
        ybase=None,
    )

    





def main():

    # time_vs_leapfrogsteps_hmc()
    # time_vs_parameters_hmc()
    leapfrog_steps_vs_parameters_nuts()
    # df = load_results()
    # df = df[
    #     df["kernel"] == "hmc"
    # ]
    # print(
    #     df[
    #         df["num_params"] == 561
    #     ]
    # )
    # print(load_results())
    return None

    


if __name__ == "__main__":
    main()