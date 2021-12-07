import numpy as np
import tensorflow as tf
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_probability as tfp
from bayesian_dense_layer import BayesianDenseLayer

import sys

np.random.seed(1)
tf.random.set_seed(1)


def get_model_bnn(layers, input_shape, activation):
    model = tf.keras.Sequential()
    model.add(
        BayesianDenseLayer(
            units=layers[0], input_shape=input_shape, activation=activation
        )
    )
    for units in layers[1:-1]:
        model.add(BayesianDenseLayer(units=units, activation=activation))
    model.add(BayesianDenseLayer(units=layers[-1], activation=None))

    return model


def get_model(layers, input_shape, activation):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(
            units=layers[0], input_shape=input_shape, activation=activation
        )
    )
    for units in layers[1:-1]:
        model.add(tf.keras.layers.Dense(units=units, activation=activation))
    model.add(tf.keras.layers.Dense(units=layers[-1], activation=None))

    return model

@tf.function
def loss_fn(model, y, yhat, lamb=1e-3):
    loss = tf.reduce_sum(tf.keras.losses.MSE(y, yhat))
    tmp = 0
    for layer in model.layers:
        tmp += tf.keras.losses.MSE(0., layer.kernel)
        tmp += tf.keras.losses.MSE(0., layer.bias)
    tmp = tf.reduce_sum(tmp)
    loss += lamb * tmp
    return loss

@tf.function
def grad_loss(model, x, y):
    with tf.GradientTape() as tape:
        yhat = model(x)
        loss = loss_fn(model, y, yhat)    
    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad

@tf.function
def get_kinetic_energy(momenta):
    K = tf.reduce_sum([tf.reduce_sum(p ** 2) for p in momenta])
    return K

@tf.function
def sample_chain(*args, **kwargs):
    return tfp.mcmc.sample_chain(*args, **kwargs)


def predict_from_chain(x_test, chain, model):
    predictions = []
    for weights in chain:
        model = build_model(weights)
        predictions.append(model(x_test).numpy())
    return predictions

@tf.function
def prior_log_prob_fn(params, lamb = 0.):
    log_prob = 0
    for w in params:
        log_prob -= tf.reduce_sum(w ** 2)
    return lamb * log_prob

def bnn_log_prob_fn(X, y, params, get_mean=False):
    """Compute log likelihood of predicted labels y given features X and params.
    Args:
        X (np.array): 2d feature values.
        y (np.array): 1d labels (ground truth).
        params (list): [[w1, b1], ...] containing 2d/1d arrays for weights/biases.
        get_mean (bool, optional): Whether to return the mean log
            probability over all labels for diagnostics, e.g. to
            compare train and test set performance.
    Returns:
        tf.tensor: Sum or mean of log probabilities of all labels.
    """
    model = build_model(params)
    #model.set_weights(params)
    yhat = model(X)

    return -0.5 * tf.reduce_sum((y - yhat)**2)

def target_log_prob_fn_factory(X_train, y_train):
    # This signature is forced by TFP's HMC kernel which calls log_prob_fn(*chains).
    def target_log_prob_fn(*params):
        log_prob = prior_log_prob_fn(params) + bnn_log_prob_fn(X_train, y_train, params)
        return log_prob
    return target_log_prob_fn


def dense(x, w, b, activation):
    return activation(tf.matmul(x, w) + b)

def build_model(params, activation=tf.nn.sigmoid):
    
    
    def model(x):
        kernel = params[::2]
        bias = params[1::2]
        for w, b in zip(kernel[:-1], bias[:-1]):
            x = activation(tf.matmul(x, w) + b)
            #x = dense(x, w, b, activation)
        x = tf.matmul(x, kernel[-1]) + bias[-1]
        return x
    return model

# def build_net(params, activation=tf.nn.relu):
#     def model(X, training=True):
#         for w, b in params[:-1]:
#             X = dense(X, w, b, activation)
#         # final linear layer
#         X = dense(X, *params[-1])
#         y_pred, y_log_var = tf.unstack(X, axis=-1)
#         y_var = tf.exp(y_log_var)
#         if training:
#             return tfp.distributions.Normal(loc=y_pred, scale=tf.sqrt(y_var))
#         return y_pred, y_var
#     return model


if __name__ == "__main__":
    
    with tf.device("/CPU:0"):
        layers = [1, 20, 1]
        #input_shape = (1,)
        #model = get_model(layers, input_shape, activation="sigmoid")
        #print(model.summary())
        f = lambda x: tf.math.sin(x)

        n_train = 1000
        dims = 1
        x_train = tf.random.normal(shape=(n_train, dims), mean=0., stddev=3.)
        y_train = f(x_train)

        #yhat = model(x_train)
        #print(yhat.shape)



        kernel_prior = tfp.distributions.Normal(loc=0., scale=0.01)
        bias_prior = tfp.distributions.Normal(loc=0., scale=0.01)

        current_state = [
            (kernel_prior.sample(sample_shape=(n, m)), bias_prior.sample(sample_shape=(m,)))
            for n, m in zip(layers[:-1], layers[1:])
        ]
        tmp = []
        for weights in current_state:
            kernel, bias = weights
            tmp.append(kernel)
            tmp.append(bias)
        current_state = tmp
        for weights in current_state:
            print(weights.shape)

        model = build_model(current_state)
        yhat = model(x_train)
        print(yhat.shape)



        target_log_prob_fn = target_log_prob_fn_factory(x_train, y_train)

        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=0.01,
            num_leapfrog_steps=60,
            store_parameters_in_results=None
        )

        # adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        #     kernel, num_adaptation_steps=num_burnin_steps
        # )

        # kernel = tfp.mcmc.NoUTurnSampler(
        #     target_log_prob_fn=target_log_prob_fn,
        #     step_size=0.01
        # )

        adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=1000,
        )

        
        num_results = 1000
        num_burnin_steps = 1000
        #current_state = model.get_weights()
        #current_state = [tf.convert_to_tensor(w) for w in current_state]

        # current_state = [np.array(w) for w in current_state]
        # current_state_shapes = [arr.shape for arr in current_state]
        chain = sample_chain(   
            kernel=kernel,
            current_state=current_state,
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=0,
            trace_fn=None,
        )
        #Sort chain according to complete networks.
        chain = [
            [tensor[i] for tensor in chain] for i in range(len(chain[0]))
        ]

        n_test = 1000
        x_test = tf.random.normal(shape=(n_test,1), mean=0., stddev=3.)
        predictions = predict_from_chain(x_test, chain, model)

        x = np.array(list(x_test.numpy().squeeze(-1))*num_results)
        predictions = np.array(predictions)
        predictions = predictions.squeeze(-1).ravel()
        sns.lineplot(x, predictions, ci="sd")
        
        x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
        plt.plot(x, f(x), label="True function", color="r")
        plt.scatter(x_train, y_train, label="observed data")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


    # with tf.device("/CPU:0"):
    #     model, samples = bayesian_fit(
    #         model,
    #         x_train,
    #         y_train,
    #         L=60,
    #         eps=0.0001,
    #         num_burnin_steps=num_burnin_steps,
    #         num_samples=num_results,
    #     )

    #     #print(f"{samples=}")
    #     n_test = 1000
    #     x_test = tf.random.normal(shape=(n_test, dims), mean=0., stddev=2.)
    #     y_test = f(x_test)

    #     predictions = []
    #     for weights in samples:
    #         model.set_weights(weights)
    #         predictions.append(model(x_test).numpy().ravel())

    #     predictions = np.array(predictions)
    #     predictions = predictions.ravel()

    #     X = np.array(list(x_test) * num_results).squeeze(-1)
    #     sns.lineplot(x=X, y=predictions, ci="sd")

    #     x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
    #     plt.plot(x, f(x), label="True function", color="r")
    #     plt.legend()
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.show()