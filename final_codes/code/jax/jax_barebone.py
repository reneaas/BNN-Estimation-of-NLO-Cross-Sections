import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.example_libraries import optimizers
from tqdm import trange
import matplotlib.pyplot as plt


def forward(x, weights, activations):
    """Computes the forward pass of a neural network.

    Args:
        x (np.ndarray):
            Numpy array of shape (num_points, num_features)
        weights (List[np.ndarray])
            List of weights of the neural network
        activations (List[Callabe]):
            List of Python callables, one per layer of the neural network
    
    Returns:
        Result of forward pass. Predictions of shape (num_points, num_outputs)
    """
    for w, b, act in zip(weights[::2], weights[1::2], activations):
        x = dense(x, w, b, act)
    return x


@partial(jax.jit, static_argnames=("activation"))
def dense(x, w, b, activation=jax.nn.tanh):
    """Dense fully-connected layer in neural network
    Helper function for the forward pass.
    """
    return activation(jnp.dot(w, x) + b)


dense = jax.vmap(dense, in_axes=(0, None, None, None))  # Vectorization of `dense`


def get_weights(layers):
    """Draws the initial weights of the neural network

    Args:
        layers (List[int]):
            List of nodes in the neural network. Structure: 
            layers = [input_features, hidden_layer_1, hidden_layer_2, ..., hidden_layer_L, output_dims]
    Returns:
        weights (list[np.ndarray]):
            List of the weights of the neural network. Structure:
            weights = [kernel_0, bias_0, kernel_1, bias_1, ..., kernel_L, bias_L]    
    """
    weights = []
    for n, m in zip(layers[1:], layers[:-1]):
        w = np.random.normal(size=(n, m)) / np.sqrt(m)
        b = np.random.normal(size=(n,))
        weights.extend([w, b])
    return weights


@jax.jit
def log_prior_fn(weights):
    """Log prior of the weights. Assumed to be a Gaussian prior
    
    Args:
        weights (List[np.ndarray]):
            List of the weights of the neural network. Structure:
            weights = [kernel_0, bias_0, kernel_1, bias_1, ..., kernel_L, bias_L] 
    Returns:
        The calculated log prior given the weights.
    """
    res = 0
    for w, b in zip(weights[::2], weights[1::2]):
        res += jnp.sum(w ** 2)
        res += jnp.sum(b ** 2)
    return 0.5 * res


def log_likelihood_fn(weights, activations, x, y):
    """Calculates the log likelihood for a set of neural network weights and data (x, y)

    Args:
        weights (List[np.ndarray]):
            List of the weights of the neural network. Structure:
            weights = [kernel_0, bias_0, kernel_1, bias_1, ..., kernel_L, bias_L] 
        activations (List[Callable]):
            List of Python callable activation functions. One per layer.
        x (np.ndarray):
            Input points of shape (num_points, num_features)
        y (np.ndarray):
            Target of shape (num_points, num_output_dims)
    Returns:
        The calculated log likelihood.    
    """
    y_pred = forward(x, weights, activations)
    log_likelihood = jnp.sum((y_pred - y) ** 2)
    return 0.5 * log_likelihood


def get_log_target_fn(x, y, lamb, likelihood_noise, activations):
    """Creates the log posterior function (the potential energy)
    of the neural network model.

    Args:
        x (np.ndarray):
            Input points of shape (num_points, num_features)
        y (np.ndarray):
            Target of shape (num_points, num_output_dims)
        lamb (float):
            Regularization strength of the prior. 
        likelihood_noise (float):
            Regularization strength of the likelihood term.
        activations (List[Callable]):
            List of Python callable activation functions. One per layer.
    Returns: 
        Python callable that calculates the log posterior function
        for a set of neural network weights.
    """

    def log_posterior_fn(weights):
        return lamb * log_prior_fn(weights) + likelihood_noise * log_likelihood_fn(
            weights, activations, x, y
        )

    return log_posterior_fn


def pretrain_network(grad_log_posterior_fn, weights, x, y, epochs, step_size=1e-3):
    """Pretrains the neural network and returns a point estimate of the weights
    given a log posterior function. 

    Args:
        grad_log_posterior_fn (Python callable):
            Gradient of the log posterior function with signature
            grad_log_posterior_fn(weights)
    
    """
    opt_init, opt_update, get_weights = optimizers.adam(step_size=step_size)
    opt_state = opt_init(weights)
    opt_update = jax.jit(opt_update)
    get_weights = jax.jit(get_weights)
    for step in trange(epochs, desc="Pretraining"):
        current_state = get_weights(opt_state)
        grads = grad_log_posterior_fn(current_state)
        opt_state = opt_update(step, grads, opt_state)
    return get_weights(opt_state)


@jax.jit
def kinetic_energy(momentum: list[np.ndarray]):
    """Calculates the kinetic energy for a set of momenta.

    Args:
        momentum (List[np.ndarray]):
            List of momenta, one for each weight in the neural network model.
    Returns:
        The calculated kinetic energy.
    """
    res = 0
    for p in momentum:
        res += jnp.sum(p ** 2)
    return 0.5 * res


def hmc_step(
    x: np.ndarray,
    y: np.ndarray,
    weights: list[np.ndarray],
    log_posterior_fn: callable,
    grad_log_posterior_fn: callable,
    num_leapfrog_steps: int,
    step_size: float,
    key: jax.random.PRNGKey,
):
    """Calculates one step with basic Hamiltonian Monte Carlo and returns a new
    set of weights.

    Args:
        x (np.ndarray):
            Input points of shape (num_points, num_features)
        y (np.ndarray):
            Target of shape (num_points, num_output_dims)
        weights (List[np.ndarray]):
            List of the weights of the neural network. Structure:
            weights = [kernel_0, bias_0, kernel_1, bias_1, ..., kernel_L, bias_L] 
        log_posterior_fn (Python callable):
            Python callable that computes the log posterior for a specified set of weights
        grad_log_posterior_fn (Python callable):
            Python callable that computes the gradient of the log posterior with respect to
            the neural network weights.
        num_leapfrog_steps (int):
            Number of Leapfrog steps to perform.
        step_size (float):
            Step size used in the Leapfrog integration
        key (jax.random.PRNGkey):
            Key used with jax's RNGs.

    Returns:
        weights (List[np.ndarray]):
            The next weights in the Markov chain.
            List of the weights of the neural network. Structure:
            weights = [kernel_0, bias_0, kernel_1, bias_1, ..., kernel_L, bias_L]
        key (jax.random.PRNGkey):
            The next key in the state of the RNG.
        is_accepted (bool):
            Boolean that is True if new state is accepted and false if rejected
            in the Metropolis correction step. 
    """
    num_vars = len(jax.tree_leaves(weights))
    all_keys = jax.random.split(key=key, num=(num_vars + 1))
    lamb = 1 if np.random.uniform() > 0.5 else -1
    init_weights = weights
    momentum = [
        jax.random.normal(key=k, shape=p.shape) for p, k in zip(weights, all_keys[1:])
    ]

    K_init = kinetic_energy(momentum)
    V_init = log_posterior_fn(weights)
    H_init = K_init + V_init

    # First update of momenta
    grads = grad_log_posterior_fn(weights)
    momentum = [p - 0.5 * lamb * step_size * dp for p, dp in zip(momentum, grads)]

    for _ in range(num_leapfrog_steps - 1):
        weights = [q + lamb * step_size * p for q, p in zip(weights, momentum)]
        grads = grad_log_posterior_fn(weights)
        momentum = [p - lamb * step_size * dp for p, dp in zip(momentum, grads)]

    grads = grad_log_posterior_fn(weights)
    momentum = [p - 0.5 * lamb * step_size * dp for p, dp in zip(momentum, grads)]

    K_final = kinetic_energy(momentum)
    V_final = log_posterior_fn(weights)
    H_final = K_final + V_final
    dH = H_final - H_init

    key, subkey = jax.random.split(key=all_keys[0])
    if dH < 0:
        is_accepted = True
        return weights, key, is_accepted
    elif jnp.exp(-dH) < jax.random.uniform(key=subkey):
        is_accepted = True
        return weights, key, is_accepted
    else:
        is_accepted = False
        return init_weights, key, is_accepted


def main():

    layers = [1, 10, 10, 10, 10, 1]
    weights = get_weights(layers=layers)
    activations = [jax.nn.tanh, jax.nn.tanh, jax.nn.tanh, jax.nn.tanh, lambda x: x]
    n_train = 2500
    x = np.random.normal(size=(n_train, 1))
    f = lambda x: x * np.sin(x) * np.cos(x)
    y = f(x)

    log_posterior_fn = get_log_target_fn(
        x=x, y=y, lamb=1e-3, likelihood_noise=1.0, activations=activations
    )

    grad_log_posterior_fn = jax.jit(jax.grad(log_posterior_fn))

    weights = pretrain_network(
        grad_log_posterior_fn=grad_log_posterior_fn,
        weights=weights,
        x=x,
        y=y,
        epochs=1000,
        step_size=1e-3,
    )

    chain = [weights]
    num_results = 1000
    seed = 1000
    key = jax.random.PRNGKey(seed)
    for _ in trange(num_results):
        weights, key, is_accepted = hmc_step(
            x=x,
            y=y,
            weights=weights,
            log_posterior_fn=log_posterior_fn,
            grad_log_posterior_fn=grad_log_posterior_fn,
            step_size=0.001,
            num_leapfrog_steps=1024,
            key=key,
        )
        chain.append(weights)

    x = np.linspace(-2, 2, 1001)
    x = x[:, None]
    predictions = [
        forward(x=x, weights=weights, activations=activations).squeeze(-1)
        for weights in chain
    ]

    x = x.squeeze(-1)
    for y_pred in predictions:
        plt.plot(x, y_pred, alpha=0.1)
    plt.plot(x, f(x), label="True", color="black")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
