import pyarma as pa
import numpy as np

def get_samples(path, num_samples, num_weights):
    path = "./bnn_samples/"
    samples = []
    for i in range(num_samples):
        weights = []
        for j in range(num_weights):
            w = pa.mat()
            fname = path + "weights" + str(i) + "_" + str(j) + ".bin"
            w.load(fname)
            w = np.array(w)
            weights.append(w)
        samples.append(weights)
    return samples

def get_samples_dict(path, num_samples, num_weights):
    pass

# def evaluate_model(x, samples):
#     predictions = []
#     for weights in samples:
#         kernel = weights[::2]
#         bias = weights[1::2]
#         x_ = np.copy(x)
#         for w, b in zip(kernel, bias):
#             x_ = np.matmul(x_, w) + b

        


num_weights = 4
num_samples = 1000

path = "./bnn_samples/"
samples = get_samples(path, num_samples, num_weights)

