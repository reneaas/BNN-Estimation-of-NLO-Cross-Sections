import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.example_libraries import optimizers
from tqdm import trange
import matplotlib.pyplot as plt
import sys


def forward(x: np.ndarray, weights: list[np.ndarray], activations: list[callable]):
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
def dense(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, activation: callable = jax.nn.tanh
):
    """Dense fully-connected layer in neural network
    Helper function for the forward pass.
    """
    return activation(jnp.dot(w, x) + b)


dense = jax.vmap(dense, in_axes=(0, None, None, None))  # Vectorization of `dense`


def get_weights(layers: list[int]):
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
def log_prior_fn(weights: list[np.ndarray]):
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


def log_likelihood_fn(
    weights: list[np.ndarray], activations: list[callable], x: np.ndarray, y: np.ndarray
):
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


def get_log_target_fn(
    x: np.ndarray,
    y: np.ndarray,
    lamb: float,
    likelihood_noise: float,
    activations: list[callable],
):
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

    def log_posterior_fn(weights: list[np.ndarray]):
        return lamb * log_prior_fn(weights) + likelihood_noise * log_likelihood_fn(
            weights, activations, x, y
        )

    return log_posterior_fn


def pretrain_network(
    grad_log_posterior_fn: callable,
    weights: list[np.ndarray],
    epochs: int,
    step_size: float = 1e-3,
):
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


def get_hmc_sampler(log_posterior_fn, grad_log_posterior_fn, num_leapfrog_steps):
    def hmc_step(weights: list[np.ndarray], step_size: float, key: jax.random.PRNGKey):
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
            jax.random.normal(key=k, shape=p.shape)
            for p, k in zip(weights, all_keys[1:])
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
        acceptance_probability = jnp.exp(-dH)
        if dH < 0:
            is_accepted = True
            return weights, key, is_accepted, acceptance_probability
        elif acceptance_probability < jax.random.uniform(key=subkey):
            is_accepted = True
            return weights, key, is_accepted, acceptance_probability
        else:
            is_accepted = False
            return init_weights, key, is_accepted, acceptance_probability

    return hmc_step


def step_size_adaptation(
    num_adaptation_steps,
    sampler,
    step_size,
    weights,
    target_accept_probability,
    kappa=0.75,
    gamma=0.05,
    t0=10,
):
    seed = 999
    key = jax.random.PRNGKey(seed)
    x = np.log10(step_size)
    x_bar = x
    mu = np.log10(10 * step_size)

    cumulative_f = 0
    for t in trange(1, num_adaptation_steps + 1):
        weights, key, is_accepted, acceptance_probability = sampler(
            weights, step_size, key
        )
        f = target_accept_probability - acceptance_probability
        cumulative_f += f
        eta = 1. / t ** kappa
        x = mu - np.sqrt(t) / (gamma * (t + t0)) * cumulative_f
        x_bar = eta * x + (1 - eta) * x_bar
        step_size = 10 ** x
    adapted_step_size = 10 ** x_bar
    return adapted_step_size


def main():

    layers = [1, 10, 1]
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
        epochs=1000,
        step_size=1e-3,
    )

    sampler = get_hmc_sampler(
        log_posterior_fn=log_posterior_fn,
        grad_log_posterior_fn=grad_log_posterior_fn,
        num_leapfrog_steps=128,
    )

    step_size = step_size_adaptation(
        num_adaptation_steps=100,
        sampler=sampler,
        step_size=1e-3,
        target_accept_probability=0.65,
        weights=weights,
    ) 

    print(f"adapted step size: {step_size = }")
    sys.exit()

    chain = [weights]
    num_results = 1000
    seed = 1000
    key = jax.random.PRNGKey(seed)
    total_accepted = 0
    for _ in trange(num_results):
        weights, key, is_accepted, acceptance_probability = sampler(
            weights=weights,
            log_posterior_fn=log_posterior_fn,
            grad_log_posterior_fn=grad_log_posterior_fn,
            step_size=0.0001,
            num_leapfrog_steps=512,
            key=key,
        )
        total_accepted += 1.0 * is_accepted
        chain.append(weights)

    accept_rate = total_accepted / num_results
    print(f"{accept_rate = }")
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
