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



def get_model(layers, train_sz, activation):
    model = tf.keras.Sequential()
    model.add(
        tfp.layers.DenseVariational(
            units=layers[0],
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1 / train_sz,
            activation=activation,
        )
    )
    for units in layers[1:-1]:
        model.add(
            tfp.layers.DenseVariational(
            units=units,
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1 / train_sz,
            activation=activation,
            )
        )
    # model.add(tf.keras.layers.Dense(units=layers[-1], activation=None))

    model.add(
        tfp.layers.DenseVariational(
            units=layers[-1],
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1 / train_sz,
            activation=None,
        )
    )

    return model

def compute_predictions(model, x, num_results):
    predictions = []
    X = []
    for i in trange(num_results, desc="Computing predictions"):
        predictions.append(model(x).numpy().squeeze(-1))
        X.append(x.numpy().squeeze(-1))
    X = np.array(X)
    predictions = np.array(predictions)
    return X, predictions


#create training data
n_train = 1000
f = lambda x: tf.math.sin(x) * tf.math.cos(x)
x_train = tf.random.normal(shape=(n_train, 1), mean=0., stddev=2.)
#x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
# x = x[:, None]
# x_train = tf.convert_to_tensor(x)
y_train = f(x_train)

layers = [100, 1]
model = get_model(layers, n_train, activation="sigmoid")
model.compile(
    optimizer="adam",
    loss="mse",
)

with tf.device("/CPU:0"):
    start = time.perf_counter()
    model.fit(x=x_train, y=y_train, batch_size=16, epochs=100)
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


