import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
import numpy as np
import time


# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model



def get_model(layers, n_train, activation):
    model = tf.keras.Sequential()
    model.add(
        tfp.layers.DenseVariational(
            units=layers[0],
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1 / n_train,
            activation=activation,
        )
    )

    # model.add(
    #     tf.keras.layers.Dense(units=layers[0], activation=activation)
    # )
    for units in layers[1:-1]:
        model.add(
            tfp.layers.DenseVariational(
            units=units,
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1 / n_train,
            activation=activation,
            )
        )

        # model.add(
        #     tf.keras.layers.Dense(units=units, activation=activation)
        # )
    # model.add(tf.keras.layers.Dense(units=layers[-1], activation=None))

    model.add(
        tfp.layers.DenseVariational(
            units=layers[-1],
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1 / n_train,
            activation=None,
        )
    )

    return model

def get_flipout_model(layers, activation):
    model = tf.keras.Sequential()
    model.add(
        tfp.layers.DenseFlipout(
            units=layers[0],
            activation=activation,
        )
    )
    for units in layers[1:-1]:
        model.add(
            tfp.layers.DenseFlipout(
            units=units,
            activation=activation,
            )
        )
    # model.add(tf.keras.layers.Dense(units=layers[-1], activation=None))
    model.add(
        tfp.layers.DenseFlipout(units=layers[-1], activation=tf.identity)
    )
    return model

def loss_fn(y_true, y_pred):
    loss = tf.reduce_mean((y_pred - y_true) ** 2)
    return loss

@tf.function
def grad_loss(x, y, model):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.reduce_mean((y - y_pred) ** 2)
        loss += tf.reduce_mean(model.losses)
    grad = tape.gradient(loss, model.trainable_variables)
    return grad, loss

def train(model, x, y, optimizer, num_epochs=100, batch_size=1000):
    losses = []
    # mse = tf.keras.losses.MSE()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if batch_size < x.shape[0]:
        ds = ds.batch(batch_size=batch_size)

    for _ in trange(num_epochs):
        for x_, y_ in ds:
            # with tf.GradientTape() as tape:
            #     y_pred = model(x_)
            #     loss = tf.reduce_mean((y_ - y_pred) ** 2)
            #     loss += tf.reduce_sum(model.losses) / x.shape[0]
            # grad = tape.gradient(loss, model.trainable_variables)
            grad, loss = grad_loss(x_, y_, model)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            # optimizer.minimize(loss=loss, var_list=model.trainable_variables)
            losses.append(loss)
    return loss

def compute_predictions(model, x, num_results):
    predictions = []
    X = []
    for i in trange(num_results, desc="Computing predictions"):
        predictions.append(model(x).numpy().squeeze(-1))
        X.append(x.numpy().squeeze(-1))
    X = np.array(X)
    predictions = np.array(predictions)
    return X, predictions

def main():
    #create training data
    n_train = 1000
    batch_size = 100
    f = lambda x: x * tf.math.sin(x) * tf.math.cos(x)
    x_train = tf.random.normal(shape=(n_train, 1), mean=0., stddev=3.)
    y_train = f(x_train) + tf.random.normal(shape=x_train.shape, mean=0.0, stddev=0.5)

    layers = [50, 1]
    model = get_model(layers, n_train=n_train, activation=tf.nn.tanh)
    # model = get_flipout_model(layers=layers, activation=tf.nn.tanh)
    model.compile(
        optimizer="adam",
        loss="mse",
        # loss=lambda y_true, y_pred: tf.reduce_mean((y_true - y_pred) ** 2) + sum(model.losses) / n_train
    )

    # model = tf.keras.Sequential([
    #     tfp.layers.DenseReparameterization(10, activation=tf.nn.sigmoid),
    #     tfp.layers.DenseReparameterization(1, activation=tf.identity),
    # ])

    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    #     loss="mse",
    # )

    start = time.perf_counter()
    # loss = train(model=model, x=x_train, y=y_train, optimizer=tf.keras.optimizers.Adam(), batch_size=10)
    model_history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=10000)
    loss = model_history.history.get("loss")
    plt.plot(loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

    # print(f"{loss=}")
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds")

    num_results = 100
    n_test = 1000
    x_test = tf.random.normal(shape=(n_test, 1), mean=0., stddev=2.)

    X, predictions = compute_predictions(model, x_test, num_results)

    sns.lineplot(X.ravel(), predictions.ravel(), ci="sd")

    x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
    plt.plot(x, f(x), label="True function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()

