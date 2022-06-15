import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rc("text", usetex=True)

from bnn.bnn import BayesianNeuralNetwork
from slha_loader.slha_loader import SLHALoader
from utils.preprocessing import split_data

def main():
    model_name = r"models/2_small_hidden_layers.npz"

    bnn = BayesianNeuralNetwork()
    bnn.load_model(fname=model_name)
    print(bnn)

    weights = bnn.weights
    print([p.shape for p in weights])

    kernels = weights[::2]
    biases = weights[1::2]

    i = 2
    j = 6
    l = 0
    x_label = r"$W_{3,7}^1$"
    y_label = r"$W_{3,6}^1$"
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
    fname = "posterior_weights2.pdf"
    plt.savefig(path + fname)
    plt.show()

    # sns.jointplot(x=weights[l][:, i, j].numpy(), y=weights[l][:, i, j-1].numpy(), kind="kde")
    # plt.xlabel(r"$W_{2,7}^0$")
    # plt.ylabel(r"$W_{2,6}^0$")



if __name__ == "__main__":
    main()