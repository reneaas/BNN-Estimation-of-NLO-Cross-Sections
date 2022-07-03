import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


gpu_std_config = {
    "num_burnin_steps": 1000,
    "batch_size": 32,
    "num_results": 1000,
    "num_steps_between_results": 10,
    "num_epochs": 2500,
}

gpu_num_samples = (
    gpu_std_config.get("num_results") * (1 + gpu_std_config.get("num_steps_between_results"))
    + gpu_std_config.get("num_burnin_steps")
)

print(f"{gpu_num_samples = }")

cpu_std_config = {
    "num_burnin_steps": 100,
    "batch_size": 32,
    "num_results": 10,
    "num_steps_between_results": 10,
    "num_burnin_steps": 0,
    "num_epochs": 0,
}

cpu_num_samples = (
    cpu_std_config.get("num_results") * (1 + cpu_std_config.get("num_steps_between_results"))
    + cpu_std_config.get("num_burnin_steps")
)


def gpu_load_results():
    fname = "./results/results_merged.pkl"
    df = pd.read_pickle(fname)
    return df

def cpu_load_results():
    fname = "./cpu_results/cpu_results_merged.pkl"
    df = pd.read_pickle(fname)
    return df



def main():
    cpu_df = cpu_load_results()
    gpu_df = gpu_load_results()
    gpu_df = gpu_df[
        gpu_df["num_leapfrog_steps"] == 512
    ]

    print(f"{gpu_df = }")
        

    # Prepare CPU data    
    cpu_data = {
        "layers": cpu_df["layers"].values,
        "timeused": cpu_df["timeused"].to_numpy(),
    }

    # Convert cpu data 
    cpu_data["timeused"] = np.array([float(x) for x in cpu_data.get("timeused")])
    cpu_data["layers"] = [eval(x) for x in cpu_data["layers"]]

    # Compute number of parameters
    cpu_data["num_params"] = []
    for layer in cpu_data.get("layers"):
        num_params = 0
        for w, b in zip(layer[:-1], layer[1:]):
            num_params += w * b + b
        cpu_data["num_params"].append(num_params)
    cpu_data["num_params"] = np.array(cpu_data.get("num_params"))
    cpu_data["layers"] = np.array(cpu_data.get("layers")) # MIGHT CHANGE THIS LATER 

    # Sort CPU data
    idx = np.argsort(cpu_data.get("num_params"))
    for key in cpu_data:
        cpu_data[key] = cpu_data.get(key)[idx]
    
    cpu_data["timeused"] /= cpu_num_samples
    print(f"{cpu_data = }")

    # Prepare GPU data

    gpu_data = {
        "layers": gpu_df["layers"].values,
        "timeused": gpu_df["timeused"].to_numpy(),
    }
    gpu_data["timeused"] = [float(x) for x in gpu_data.get("timeused")]
    gpu_data["layers"] = [list(x) for x in gpu_data.get("layers")]
    # print(gpu_data)
    layers = []
    time = []
    for i, layer in enumerate(gpu_data.get("layers")):
        if len(layer) == 3 and layer[1] < 2048:
            layers.append(layer)
            time.append(gpu_data.get("timeused")[i])
    gpu_data = {
        "layers": layers,
        "timeused": np.array(time)
    }

    # Compute number of parameters
    gpu_data["num_params"] = []
    for layer in gpu_data.get("layers"):
        num_params = 0
        for w, b in zip(layer[:-1], layer[1:]):
            num_params += w * b + b
        gpu_data["num_params"].append(num_params)
    gpu_data["num_params"] = np.array(gpu_data.get("num_params"))

    # Sort the GPU data
    idx = np.argsort(gpu_data.get("num_params"))
    gpu_data["layers"] = np.array(gpu_data.get("layers"))
    # gpu_data["layers"] = [np.array(x) for x in gpu_data.get("layers")]
    # gpu_data["layers"] = np.array(gpu_data.get("layers"), dtype=object)
    # gpu_data["layers"] = gpu_data["layers"][idx]
    print(f"{gpu_data = }")
    for key in gpu_data:
        gpu_data[key] = gpu_data.get(key)[idx]
    gpu_data["timeused"] /= gpu_num_samples
    print(f"{gpu_data = }")



    nodes = [2 ** i for i in range(5, 11)]
    # Compute relative times with CPU as basel
    data = {"num_params": [], "relative_time": []}
    for i, (cpu_time, gpu_time) in enumerate(zip(cpu_data.get("timeused"), gpu_data.get("timeused"))):
        data["num_params"].append(cpu_data.get("num_params")[i])
        data["relative_time"].append(cpu_time / gpu_time)
    print(f"{data = }")

    plt.plot(nodes, data.get("relative_time"))
    plt.scatter(nodes, data.get("relative_time"), color="red")
    plt.xlabel("Number of Hidden Layer Nodes")
    plt.ylabel("Relative Time")
    plt.xscale("log", base=2)

    path = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/cpu_vs_gpu/"
    fname = "cpu_vs_gpu_performance.pdf"
    plt.savefig(path + fname)
    # plt.show()
    plt.close()


    plt.scatter(nodes, gpu_data.get("timeused"), color="red")
    plt.plot(nodes, gpu_data.get("timeused"))
    plt.xlabel("Number of Hidden Layer Nodes")
    plt.ylabel("Time Used per Sample [s]")
    plt.xscale("log", base=2)

    fname = "gpu_training_time.pdf"
    plt.savefig (path + fname)
    plt.close()




if __name__ == "__main__":
    main()

