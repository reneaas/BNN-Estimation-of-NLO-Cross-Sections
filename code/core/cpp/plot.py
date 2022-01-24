import pyarma as pa
import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return x * 1.*(x > 0)

def sigmoid(x):
    return 1./(1 + np.exp(-x))

def forward(x, kernel, bias):
    for w, b in zip(kernel[:-1], bias[:-1]):
        x = relu(np.matmul(x, w) + b)
    x = np.matmul(x, kernel[-1]) + bias[-1]


def main():
    weights = []
    for i in range(6):
        param = pa.cube()
        param.load(f"models/weight:{i}")
        weights.append(np.array(param))
    print(weights[3].shape)

    x_train = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
    x_train = x_train[:, None]
    kernel = weights[::2]
    kernel = [
        w.reshape(w.shape[0], w.shape[-1], w.shape[1]) for w in kernel
    ]
    bias = weights[1::2]
    bias = [
        b.reshape(b.shape[0], b.shape[-1], b.shape[1]) for b in bias
    ]
    x = np.zeros_like(x_train)
    x[:] = x_train[:]
    for w, b in zip(kernel[:-1], bias[:-1]):
        x = sigmoid(np.matmul(x, w) + b)
    x = np.matmul(x, kernel[-1]) + bias[-1]
    for i in range(x.shape[0]):
        plt.plot(x_train[:, 0], x[i].ravel())
    plt.show()



if __name__ == "__main__":
    main()
