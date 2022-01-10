import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bnn import BayesianNeuralNetwork
tf.random.set_seed(100)


def main():
    fname = "models/no_u_turn_adaptive.npz"
    bnn = BayesianNeuralNetwork()
    bnn.load_model(fname=fname)
    x = tf.random.normal(shape=(1,1))
    f = lambda x: x * tf.math.sin(x) * tf.math.cos(x)
    y_true = f(x)
    y_pred = bnn(x).numpy().squeeze(-1)
    print(y_pred)
    print(y_pred.shape)
    y_pred_mean = np.mean(y_pred, axis=0)
    # plt.hist(y_pred)
    sns.histplot(y_pred)
    plt.axvline(y_true.numpy().squeeze(-1), color="red", label="True value")
    plt.axvline(y_pred_mean, color="green", label="Predicted mean")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

    
    