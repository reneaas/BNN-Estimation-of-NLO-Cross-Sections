from slha_loader.slha_loader import SLHAloader
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import isnan
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import time


def get_dataset(feat_dir, target_dir, ids):
    dl = SLHAloader(ids, feat_dir, target_dir)
    dl.to_numpy()
    features = dl.features
    targets = dl.targets["nlo"]

    # Remove ill datapoints. Marked by -1 and nan.
    features = features[targets != (-1)]
    targets = targets[targets != (-1)]
    idx = np.isnan(targets) == False
    targets = targets[idx]
    features = features[idx]

    features = tf.convert_to_tensor(features, dtype=tf.float32)
    targets = tf.convert_to_tensor(targets, dtype=tf.float32)
    features, targets = scale_data(features, targets)
    return features, targets


def create_dataset(features, targets, batch_size=16):
    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size=batch_size)
    return ds


def split_data(features, targets, p_train=0.80, p_val=0.10, p_test=0.1):
    num_points = len(features)
    x_train = features[: int(num_points * p_train)]
    y_train = targets[: int(num_points * p_train)]

    x_val = features[int(num_points * p_train) : int(num_points * (p_train + p_val))]
    y_val = targets[int(num_points * p_train) : int(num_points * (p_train + p_val))]

    x_test = features[int(num_points * (p_train + p_val)) :]
    y_test = x_test = features[int(num_points * (p_train + p_val)) :]
    return x_train, y_train, x_val, y_val, x_test, y_test


def scale_data(x, y):
    x_mean = tf.math.reduce_mean(x, axis=1)
    x_std = tf.math.reduce_std(x, axis=1)
    x = (x - x_mean[..., None]) / x_std[..., None]
    y_mean = tf.math.reduce_mean(y)

    y = tf.math.log(y)
    #y_std = tf.math.reduce_std(y)
    #y = (y - y_mean[..., None]) / y_std[..., None]
    return x, y


def get_model(optimizer, loss_fn, metrics):
    model = keras.Sequential(
        [
            layers.Dense(50, activation="relu", input_shape=(1,)),
            #layers.Dense(50, activation="relu"),
            # layers.Dropout(rate=0.2),
            #layers.Dense(1000, activation="relu"),
            layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model


def fit(model, x_train, y_train, x_val, y_val):
    history = model.fit(
            x_train, y_train, batch_size=16, epochs=100, validation_data=(x_val, y_val)
    )
    return model, history

def get_bnn(optimizer, loss_fn):
    model = tf.keras.Sequential([
        tfp.layers.DenseFlipout(512, activation=tf.nn.relu, input_shape=(10,)),
        tfp.layers.DenseFlipout(10),
    ])
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model




if __name__ == "__main__":
    ids = ["1000022", "1000022"]
    target_dir = "./targets"
    feat_dir = "./features"
    features, targets = get_dataset(feat_dir, target_dir, ids)
    # ds = create_dataset(features, targets, batch_size=16)
    # optimizer = keras.optimizers.Adam()
    # loss = (keras.losses.MeanSquaredError(),)
    # model = get_model(optimizer, loss)
    # print(model.summary())
    # print(model(features[0]))

    n_train = 1000
    f = lambda x: tf.math.sin(x) * tf.math.cos(x)
    x_train = tf.random.normal(shape=(n_train, 1))
    y_train = f(x_train)


    # optimizer = keras.optimizers.RMSprop(learning_rate = 0.001),
    # optimizer = keras.optimizers.SGD(learning_rate = 0.1)
    optimizer = keras.optimizers.Adam()
    loss = (keras.losses.MeanSquaredError(),)
    metrics = [keras.metrics.MeanSquaredError()]
    model = get_model(optimizer, loss, metrics)
    model.summary()
    #x_train, y_train, x_val, y_val, x_test, y_test = split_data(features, targets)
    #model, history = fit(model, x_train, y_train, x_val, y_val)
    #model, history = fit(model, x_train, y_train)
    with tf.device("/CPU:0"):
        start = time.perf_counter()
        model.fit(x_train, y_train, epochs = 10000, batch_size=1000)
        end = time.perf_counter()
        timeused = end - start
        print(f"{timeused=} seconds ")
    # plt.plot(history.history["loss"], label = "training")
    # plt.plot(history.history["val_loss"], label = "validation")
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.legend()
    # plt.show()
