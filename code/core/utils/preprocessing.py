import numpy as np
import tensorflow as tf


def split_data(features, targets, train=0.8, val=0.05):
    test = 1 - train - val
    num_points = features.shape[0]
    features_train = features[: int(train * num_points), ...]
    targets_train = targets[: int(train * num_points), ...]

    features_val = features[
        int(train * num_points) : int((train + val) * num_points), ...
    ]
    targets_val = targets[
        int(train * num_points) : int((train + val) * num_points), ...
    ]

    features_test = features[int((train + val) * num_points) :, ...]
    targets_test = targets[int((train + val) * num_points) :, ...]

    data = {
        "train": (features_train, targets_train),
        "val": (features_val, targets_val),
        "test": (features_test, targets_test)
    }

    return data

if __name__ == "__main__":
    x = np.random.normal(size=(100, 1))
    y = x * np.sin(x) * np.cos(x)

    data = split_data(x, y)
    print(data)
