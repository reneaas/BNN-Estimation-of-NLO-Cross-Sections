import haiku as hk
import jax.numpy as jnp
import jax
from jax.example_libraries import optimizers
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
from functools import partial


def log_likelihood(x, y):
    model = hk.Sequential(
        [
            hk.Linear(10),
            jax.nn.relu,
            hk.Linear(10),
            jax.nn.relu,
            hk.Linear(10),
            jax.nn.relu,
            hk.Linear(10),
            jax.nn.relu,
            hk.Linear(1),
        ]
    )
    y_pred = model(x)
    return 0.5 * jnp.sum((y - y_pred) ** 2)


def forward(x):
    model = hk.Sequential(
        [
            hk.Linear(10),
            jax.nn.relu,
            hk.Linear(10),
            jax.nn.relu,
            hk.Linear(10),
            jax.nn.relu,
            hk.Linear(10),
            jax.nn.relu,
            hk.Linear(1),
        ]
    )
    return model(x)


def update_rule(param, grad, lr=1e-6):
    return param - grad * lr


def mle_fit(grad_log_prior, grad_log_likelihood, params, x, y, epochs, step_size=1e-3):
    opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
    opt_state = opt_init(params)
    opt_update = jax.jit(opt_update)
    get_params = jax.jit(get_params)
    for step in trange(epochs, desc="Epochs (MLE fit)"):
        current_state = get_params(opt_state)
        grads = jax.tree_multimap(
            lambda g1, g2: g1 + g2,
            grad_log_prior(current_state),
            grad_log_likelihood(current_state, x, y),
        )
        opt_state = opt_update(step, grads, opt_state)
    return get_params(opt_state)


@jax.jit
def kinetic_energy(momentum):
    return 0.5 * sum([jnp.sum(p ** 2) for p in jax.tree_leaves(momentum)])


@jax.jit
def log_prior(params, lamb=1e-3):
    return 0.5 * lamb * sum([jnp.sum(w ** 2) for w in jax.tree_leaves(params)])

@partial(jax.jit, static_argnames=("log_likelihood_fn"))
def potential_energy(params, log_likelihood_fn, x, y, lamb=1e-3):
    return 0.5 * lamb * sum(
        [jnp.sum(w ** 2) for w in jax.tree_leaves(params)]
    ) + log_likelihood_fn.apply(params, x, y)

# @partial(
#     jax.jit, static_argnames=(
#         "log_likelihood",
#         "log_prior",
#         "grad_log_likelihood",
#         "grad_log_prior",
#         "leapfrog_steps",
#         "step_size",
#     )
# )


def hmc_step(
    x,
    y,
    params,
    log_likelihood,
    log_prior,
    grad_log_likelihood,
    grad_log_prior,
    leapfrog_steps,
    step_size,
    key,
):
    treedef = jax.tree_structure(params)
    num_vars = len(jax.tree_leaves(params))
    all_keys = jax.random.split(key=key, num=(num_vars + 1))
    lamb = 1 if np.random.uniform() > 0.5 else -1

    init_params = params

    # Create initial momentum
    momentum = jax.tree_multimap(
            lambda p, k: jax.random.normal(key=k, shape=p.shape),
            params,
            jax.tree_unflatten(treedef, all_keys[1:]),
        )

    # Get initial kinetic and potential energy
    K_init = kinetic_energy(momentum)
    V_init = jax.tree_multimap(
        lambda val1, val2: val1 + val2,
        log_prior(params),
        log_likelihood(params, x, y),
    )
    E_init = K_init + V_init

    # First update of momenta
    grads = jax.tree_multimap(
        lambda g1, g2: g1 + g2,
        grad_log_prior(params),
        grad_log_likelihood(params, x, y),
    )
    momentum = jax.tree_multimap(
        lambda p, dp: p - 0.5 * lamb * step_size * dp, momentum, grads
    )

    # First update of positions
    for _ in range(leapfrog_steps - 1):
        params = jax.tree_multimap(
            lambda q, p: q + lamb * step_size * p, params, momentum
        )
        grads = jax.tree_multimap(
            lambda g1, g2: g1 + g2,
            grad_log_prior(params),
            grad_log_likelihood(params, x, y),
        )
        momentum = jax.tree_multimap(
            lambda p, dp: p - lamb * step_size * dp, momentum, grads
        )

    # Final update of positions and momentum
    grads = jax.tree_multimap(
        lambda g1, g2: g1 + g2,
        grad_log_prior(params),
        grad_log_likelihood(params, x, y),
    )
    momentum = jax.tree_multimap(
        lambda p, dp: p - 0.5 * lamb * step_size * dp, momentum, grads
    )

    # Finally reverse momentum
    # momentum = jax.tree_multimap(lambda p: -p, momentum)

    K_final = kinetic_energy(momentum)
    V_final = jax.tree_multimap(
        lambda val1, val2: val1 + val2,
        log_prior(params),
        log_likelihood(params, x, y),
    )

    E_final = K_final + V_final
    dE = E_final - E_init

    key, subkey = jax.random.split(key=all_keys[0])
    if dE < 0:
        is_accepted = True
        return params, key, is_accepted
    elif jnp.exp(-dE) < jax.random.uniform(key=subkey):
        is_accepted = True
        return params, key, is_accepted
    else:
        is_accepted = False
        return init_params, key, is_accepted



def main():
    log_likelihood_fn = hk.transform(log_likelihood)
    log_likelihood_fn = hk.without_apply_rng(log_likelihood_fn)
    rng = jax.random.PRNGKey(42)

    n_train = 10000
    x = np.random.normal(size=(n_train, 1), loc=0.0, scale=2.0)
    f = lambda x: x * np.cos(x) * np.sin(x)
    y = f(x)
    params = log_likelihood_fn.init(rng, x, y)

    grad_log_likelihood_fn = jax.jit(jax.grad(log_likelihood_fn.apply))
    grad_log_prior_fn = jax.jit(jax.grad(log_prior))
    start = time.perf_counter()
    params = mle_fit(
        grad_log_prior=grad_log_prior_fn,
        grad_log_likelihood=grad_log_likelihood_fn,
        params=params,
        x=x,
        y=y,
        epochs=1000,
        step_size=0.001,
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds on MLE fit.")

    start = time.perf_counter()
    seed = 1000
    key = jax.random.PRNGKey(seed)
    burnin_steps = 10
    for _ in trange(burnin_steps, desc="Burn-in progress"):
        params, key, is_accepted = hmc_step(
            x=x,
            y=y,
            params=params,
            log_likelihood=log_likelihood_fn.apply,
            log_prior=log_prior,
            grad_log_likelihood=grad_log_likelihood_fn,
            grad_log_prior=grad_log_prior_fn,
            leapfrog_steps=1024,
            step_size=0.0001,
            key=key,
        )

    num_results = 10
    chain = []
    num_accepted = 0
    for _ in trange(num_results, desc="Sampling"):
        params, key, is_accepted = hmc_step(
            x=x,
            y=y,
            params=params,
            log_likelihood=log_likelihood_fn.apply,
            log_prior=log_prior,
            grad_log_likelihood=grad_log_likelihood_fn,
            grad_log_prior=grad_log_prior_fn,
            leapfrog_steps=1024,
            step_size=0.0001,
            key=key,
        )
        num_accepted += 1.0 * is_accepted
        chain.append(params)
    print("acceptance ratio = ", num_accepted / num_results)
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds on sampling.")

    forward_fn = hk.transform(forward)
    forward_fn = hk.without_apply_rng(forward_fn)
    predictions = []
    x_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    x_test = x_test[:, None]
    for weights in chain:
        tmp = forward_fn.apply(weights, x_test)
        plt.plot(x_test, tmp)
        predictions.append(tmp)
    plt.show()

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    stddev = np.std(predictions, axis=0)

    plt.plot(x_test, mean_prediction, label="mean prediction", color="r")
    plt.fill_between(
        x=x_test.squeeze(-1),
        y1=(mean_prediction - stddev).squeeze(-1),
        y2=(mean_prediction + stddev).squeeze(-1),
        alpha=0.5,
    )
    plt.plot(x_test, f(x_test), label="True function", color="g")
    plt.show()

    y_test = f(x_test).squeeze(-1)
    print(y_test.shape)
    print(mean_prediction.shape)

    rel_error = (y_test - mean_prediction.squeeze(-1)) / y_test
    print(rel_error.shape)
    rel_error = np.array(rel_error)
    print(rel_error.shape)
    sns.histplot(rel_error)
    # plt.show()

    # momenta = []
    # for layer in params:
    #     for param in layer:
    #         key, subkey = jax.random.split(key)
    #         momenta.append(
    #             jax.random.normal(key=subkey, shape=layer.get(param).shape)
    #         )

    # predict = hk.transform(predict_vals)
    # predict = hk.without_apply_rng(predict)
    # predict = jax.jit(predict.apply)

    # n_test = 1000
    # x_test = np.linspace(-2 * np.pi, 2 * np.pi, 1001)
    # y_pred = predict(params=params, x=x_test[:, None])
    # y_pred = np.asarray(y_pred)

    # plt.plot(x_test, y_pred.squeeze(-1))
    # plt.show()


if __name__ == "__main__":
    main()
