from slha_loader.slha_loader import SLHAloader
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import isnan
import numpy as np

def get_dataset(feat_dir, target_dir, ids):
    dl = SLHAloader(ids, feat_dir, target_dir)
    dl.to_numpy()
    features = dl.features
    targets = dl.targets["nlo"]
    features = features[targets != (-1)]
    targets = targets[targets != (-1)]
    idx = np.isnan(targets) == False
    targets = targets[idx]
    features = features[idx]

    #Remove ill datapoints. Targets are labeled with -1 if they are to be removed.
    features = tf.convert_to_tensor(features, dtype = tf.float32)
    targets = tf.convert_to_tensor(targets, dtype = tf.float32)

    #dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    #dataset = dataset.batch(batch_size = 16)
    return features, targets

def split_data(features, targets, p_train = 0.85, p_val = 0.05, p_test = 0.1):
    num_points = len(features)
    x_train = features[:int(num_points*p_train)]
    y_train = targets[:int(num_points*p_train)]

    x_val = features[int(num_points*p_train) : int(num_points*(p_train + p_val))]
    y_val = targets[int(num_points*p_train) : int(num_points*(p_train + p_val))]

    x_test = features[int(num_points*(p_train + p_val)):]
    y_test = x_test = features[int(num_points*(p_train + p_val)):]
    return x_train, y_train, x_val, y_val, x_test, y_test

def scale_data(x, y):
    x_mean = tf.math.reduce_mean(x, axis = 1)
    x_std = tf.math.reduce_std(x, axis = 1)
    x = (x - x_mean[..., None])/x_std[..., None]
    #x /= x_std
    y_mean = tf.math.reduce_mean(y)
    y_std = tf.math.reduce_std(y)
    y = (y - y_mean[..., None])/y_std[..., None]
    return x, y



def get_model():
    model = keras.Sequential(
        [
            layers.Dense(1000, activation="sigmoid", input_shape=(10,)),
            layers.Dense(1000, activation="relu"),
            #layers.Dense(1)
            layers.Dense(1000, activation="linear")
        ]
    )
    #model.build()
    model.compile(
        #optimizer = keras.optimizers.Adam(lr = 1e-8),
        optimizer = keras.optimizers.RMSprop(),
        loss = keras.losses.MeanSquaredError(),
        metrics = [keras.metrics.MeanSquaredError()]
    )
    return model

# def get_model():
#     model = keras.Sequential(
#         [
#             layers.Dense(1000, activation="relu", input_shape=(784,)),
#             layers.Dense(1000, activation="relu"),
#             #layers.Dense(1)
#             layers.Dense(10, activation="softmax")
#         ]
#     )
#     #model.build()
#     model.compile(
#         optimizer=keras.optimizers.RMSprop(),  # Optimizer
#         # Loss function to minimize
#         loss=keras.losses.SparseCategoricalCrossentropy(),
#         # List of metrics to monitor
#         metrics=[keras.metrics.SparseCategoricalAccuracy()],
#     )
#     return model

if __name__ == "__main__":
    ids = ["1000022", "1000023"]
    target_dir = "./targets"
    feat_dir = "./features"
    features, targets = get_dataset(feat_dir, target_dir, ids)
    model = get_model()
    model.summary()
    #print(features[0])
    features, targets = scale_data(features, targets)
    #print(features[0])
    #print(tf.math.reduce_mean(features[0]))
    #print(tf.math.reduce_std(features[0]))
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(features, targets)
    model.fit(
        x_train,
        y_train,
        batch_size = 16,
        epochs = 100,
        validation_data = (x_val, y_val)
    )

    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    #
    # # Preprocess the data (these are NumPy arrays)
    # x_train = x_train.reshape(60000, 784).astype("float32") / 255
    # x_test = x_test.reshape(10000, 784).astype("float32") / 255
    #
    # y_train = y_train.astype("float32")
    # y_test = y_test.astype("float32")
    #
    # # Reserve 10,000 samples for validation
    # x_val = x_train[-10000:]
    # y_val = y_train[-10000:]
    # x_train = x_train[:-10000]
    # y_train = y_train[:-10000]
    # model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val))
