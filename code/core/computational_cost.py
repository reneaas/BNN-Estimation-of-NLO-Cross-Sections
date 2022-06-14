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

def plot_data(x, y, xlabel, ylabel, log_scale=True, xbase=None, ybase=None):
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
    plot_data(
        x=x, 
        y=y, 
        xlabel="Number of Leapfrog steps",
        ylabel="Seconds per Sample",
        xbase=2,
        ybase=None,
    )
    print(x, y)
    total_time = y * num_samples
    print(total_time)

def time_vs_parameters_hmc():
    df = load_results()
    for key in std_config:
        df = df[df[key] == std_config.get(key)]
    kernel = "hmc"
    df = df[df["kernel"] == kernel]
    num_leapfrog_steps = 512 
    df = df[df["num_leapfrog_steps"] == num_leapfrog_steps]

    x = df["num_params"]
    y = df["timeused"]
    print(df)
    fig = plot_data(
        x=x,
        y=y,
        xlabel="Number of Parameters",
        ylabel="Time used in seconds",
        log_scale=True,
    )


def leapfrog_steps_vs_parameters_nuts():
    df = load_results()
    kernel = "nuts"
    df = df[df["kernel"] == kernel]
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
    time_vs_leapfrogsteps_hmc()
    # time_vs_parameters_hmc()
    # leapfrog_steps_vs_parameters_nuts()

    # print(load_results())
    return None
    


if __name__ == "__main__":
    main()