import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rc("text", usetex=True)
import sys

from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data
sns.set(font_scale=1.5)


def main():
    # model_name = r"models/2_small_hidden_layers.npz"
    model_name = "models/4_hidden_layers_tanh.npz"

    bnn = BayesianNeuralNetwork()
    bnn.load_model(fname=model_name)
    print(bnn)

    weights = bnn.weights

    kernels = weights[::2]
    biases = weights[1::2]
    print([p.shape for p in biases])

    weights_data = {
        "kernels": kernels,
        "biases": biases,
        "ids": [
            [(0, 1, 4), (0, 1, 5)],
            [(2, 10, 3), (1, 7, 0)],
            [(1, 5), (3, 0)],
            [(2, 10), (0, 5)],
        ],
        "labels": [
            (r"$W_{2, 5}^1$", r"$W_{2, 6}^1$"), 
            (r"$W_{11, 4}^3$", r"$W_{8, 1}^1$"), 
            (r"$b_6^2$", r"$b_1^4$"),
            (r"$b_{11}^3$", r"$b_{1}^6$"),
        ],
    }

    dir = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/posterior_distribution/"

    for n, (ids, labels) in enumerate(zip(weights_data.get("ids"), weights_data.get("labels"))):

        first_axis, second_axis = ids
        xlabel, ylabel = labels
        if len(first_axis) == 3:
            lx, ix, jx = first_axis
            ly, iy, jy = second_axis
            data = weights_data.get("kernels") 

            df = pd.DataFrame(
                {
                    xlabel: data[lx][:, ix, jx],
                    ylabel: data[ly][:, iy, jy],
                }
            )
        else:
            lx, jx = first_axis
            ly, jy = second_axis
            data = weights_data.get("biases")
            df = pd.DataFrame(
                {
                    xlabel: data[lx][:, jx],
                    ylabel: data[ly][:, jy],
                }
            )
    

        s = sns.jointplot(data=df, x=xlabel, y=ylabel, kind="kde")
        fname = dir + f"posterior_weights_{n}.pdf"
        plt.savefig(fname)
        plt.close()
        # s.set_xlabel(xlabel, fontsize=25)

        
    sys.exit()
    i = 1
    j = 4
    l = 0
    x_label = r"$W_{2,5}^1$"
    y_label = r"$W_{2,4}^1$"
    w = weights[l][:, i, j].numpy().ravel()
    # sns.histplot(w)
    plt.hist(w, bins=100, histtype="step")
    plt.show()


    # 2D-distribution plots.
    # sns.displot(x=weights[l][:, i, j].numpy(), y=weights[l][:, i, j-1].numpy(), kind="kde")
    sns.kdeplot(x=kernels[l][:, i, j].numpy(), y=kernels[l][:, i, j-1].numpy())
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    data = {
        x_label: kernels[l][:, i, j].numpy(),
        y_label: kernels[l][:, i, j-1].numpy()
    }

    df = pd.DataFrame(data)
    sns.jointplot(data=df, x=x_label, y=y_label, kind="kde")
    path = "/Users/reneaas/Documents/skole/master/thesis/master_thesis/tex/thesis/figures/posterior_distribution/"
    fname = "posterior_weights1.pdf"
    plt.savefig(path + fname)
    plt.show()

    # sns.jointplot(x=weights[l][:, i, j].numpy(), y=weights[l][:, i, j-1].numpy(), kind="kde")
    # plt.xlabel(r"$W_{2,7}^0$")
    # plt.ylabel(r"$W_{2,6}^0$")



if __name__ == "__main__":
    main()