import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    i = 2
    j = 3
    l = 0
    w = weights[l][:, i, j].numpy().ravel()
    # sns.histplot(w)
    plt.hist(w, bins=100, histtype="step")
    plt.show()


    # 2D-distribution plots.
    sns.displot(x=weights[l][:, i, j].numpy(), y=weights[l][:, i, j-1].numpy(), kind="kde")
    plt.show()



if __name__ == "__main__":
    main()