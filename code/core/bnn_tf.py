import numpy as np
import tensorflow as tf
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(100)
tf.random.set_seed(100)


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
def loss_fn(model, y, yhat, lamb=1e-5):
    loss = tf.keras.losses.MSE(y, yhat)
    tmp = 0
    for layer in model.layers:
        tmp += tf.nn.l2_loss(layer.kernel)
        tmp += tf.nn.l2_loss(layer.bias)
    loss += lamb * tmp
    return loss


@tf.function
def grad_loss(model, x, y):
    with tf.GradientTape() as tape:
        yhat = model(x)
        loss = loss_fn(model, y, yhat)
        loss = tf.reduce_mean(loss)

    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad


def get_kinetic_energy(momenta):
    K = tf.reduce_sum([tf.reduce_sum(p ** 2) for p in momenta])
    return K


def bayesian_fit(model, x, y, L, eps, num_burnin_steps, num_samples):
    num_accepted = 0
    for i in trange(num_burnin_steps, desc="Burn in iterations"):
        #current_state = model.get_weights()
        #current_state = [tf.convert_to_tensor(w) for w in current_state]
        current_state = model.get_weights()
        model, accepted = hmc_step(model, x, y, L, eps, current_state)

    samples = []
    for i in trange(num_samples, desc="Generating samples"):
        #current_state = model.get_weights()
        #current_state = [tf.convert_to_tensor(w) for w in current_state]
        current_state = model.get_weights()
        model, accepted = hmc_step(model, x, y, L, eps, current_state)
        if accepted:
            num_accepted += 1
        samples.append(model.get_weights())
    
    print(f"Accepted states = {num_accepted}/{num_samples}")

    return model, samples


#@tf.function
def hmc_step(model, x, y, L, eps, current_state):

    # Initialize momenta for the kernels and biases
    momenta = [tf.random.normal(w.shape) for w in current_state]
    K_init = get_kinetic_energy(momenta)

    # Compute initial potential energy (loss) and gradient.
    loss_init, grad = grad_loss(model, x, y)

    # First step of leapfrog
    momenta = [p - 0.5 * eps * dp for p, dp in zip(momenta, grad)]

    # Inner steps of leapfrog iterations
    weights = [w for w in current_state]
    for i in range(L - 1):
        # Update network parameters
        weights = [w + eps * p for w, p in zip(weights, momenta)]
        model.set_weights(weights)

        # Compute new gradients
        loss, grad = grad_loss(model, x, y)

        momenta = [p - 0.5 * eps * dp for p, dp in zip(momenta, grad)]

    # Final step in leapfrog
    weights = [w + eps * p for w, p in zip(weights, momenta)]
    model.set_weights(weights)
    #model.weights = weights
    loss, grad = grad_loss(model, x, y)

    #Final update of momenta 
    momenta = [p - 0.5 * eps * dp for p, dp in zip(momenta, grad)]

    # Get final energies and energy differences.
    K_final = get_kinetic_energy(momenta)
    dK = K_final - K_init
    dV = loss - loss_init
    dE = dK + dV
    dE = tf.reduce_mean(dE)

    if not np.random.uniform() <= min(1, np.exp(-dV) * np.exp(-dK)):
        model.set_weights(current_state)
        accepted = False
    else:
        accepted = True

    return model, accepted


if __name__ == "__main__":
    layers = [50, 1]
    input_shape = (1,)
    model = get_model(layers, input_shape, activation="sigmoid")
    print(model.summary())
    f = lambda x: np.sin(x)
    n_train = 100
    dims = 1
    x_train = tf.random.normal(shape=(n_train, dims), mean=0., stddev=2.)
    y_train = f(x_train)

    num_samples = 1000
    num_burnin_steps = 1000

    with tf.device("/CPU:0"):
        model, samples = bayesian_fit(
            model,
            x_train,
            y_train,
            L=40,
            eps=0.01,
            num_burnin_steps=num_burnin_steps,
            num_samples=num_samples,
        )

        #print(f"{samples=}")
        n_test = 1000
        x_test = tf.random.normal(shape=(n_test, dims), mean=0., stddev=2.)
        y_test = f(x_test)

        predictions = []
        for weights in samples:
            model.set_weights(weights)
            predictions.append(model(x_test).numpy().ravel())

        predictions = np.array(predictions)
        predictions = predictions.ravel()

        X = np.array(list(x_test) * num_samples).squeeze(-1)
        sns.lineplot(x=X, y=predictions, ci="sd")

        x = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
        plt.plot(x, f(x), label="True function", color="r")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()