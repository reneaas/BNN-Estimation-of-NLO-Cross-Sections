import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import sys
from tqdm import trange
import time

# @partial(jax.jit, static_argnames=("weights", "activations"))
def forward(x, weights, activations):
    for w, b, act in zip(weights[::2], weights[1::2], activations):
        x = dense(x, w, b, act)
    return x

@partial(jax.jit, static_argnames=("activation"))
def dense(x, w, b, activation=jax.nn.tanh):
    return activation(jnp.dot(w, x) + b)
dense = jax.vmap(dense, in_axes=(0, None, None, None))

def loss_fn(x, y, weights, activations):
    x = jax.tree_multimap(lambda w: forward(weights=w, x=x, activations=activations), weights)
    return jnp.mean((x - y) ** 2)

def get_leapfrog_integrator(grad_V_fn):
    def leapfrog(params, momenta, step_size):
        grads = grad_V_fn(params)
        momenta = [p - 0.5 * step_size * dp for p, dp in zip(momenta, grads)]
        params = [q + step_size * p for q, p in zip(params, momenta)]

        grads = grad_V_fn(params)
        momenta = [p - 0.5 * step_size * dp for p, dp in zip(momenta, grads)]
        return params, momenta
    return leapfrog

def get_potential_energy_fn(x, y, forward_fn):
    @jax.jit
    def V(q, lamb=1e-3):
        l2_err = 0
        y_hat = forward_fn(x=x, q=q)
        l2_err += 0.5 * jnp.sum((y_hat - y) ** 2)

        reg_term = 0
        for w, b in zip(q[::2], q[1::2]):
            reg_term += jnp.sum(w ** 2)
            reg_term += jnp.sum(b ** 2)

        reg_term *= 0.5 * lamb
        return reg_term + l2_err
    return V
        
        

        
def hmc_step(params, V_fn, K_fn, num_leapfrog_steps, step_size):
    momenta = [np.random.normal(size=q.shape) for q in params]
    init_params = [q for q in params]

    K_init = K_fn(momenta)
    V_init = V_fn(params)
    H_init = K_init + V_init

    grad_V_fn = jax.grad(V_fn, argnums=(0))
    leapfrog = get_leapfrog_integrator(grad_V_fn=grad_V_fn)

    for _ in range(num_leapfrog_steps):
        params, momenta = leapfrog(params=params, momenta=momenta, step_size=step_size)
    K_final = K_fn(momenta)
    V_final = V_fn(params)
    H_final = K_final + V_final

    dH = H_final - H_init
    if dH < 0:
        return params
    elif np.exp(-dH) >= np.random.uniform():
        return params
    else:
        return init_params


@jax.jit
def kintetic_energy(p):
    K = 0
    for x in p:
        K += jnp.sum(x ** 2)
    return 0.5 * K


def get_weights(layers):
    weights = []
    for n, m in zip(layers[1:], layers[:-1]):
        w = np.random.normal(size=(n, m)) / np.sqrt(m)
        b = np.random.normal(size=(n,))
        weights.extend([w, b])
    return weights

def update_fn(param, grad, lr=0.001):
    return param - grad * lr




def main():

    layers = [1, 2, 1]
    params = get_weights(layers=layers)
    activations = [jax.nn.tanh, jax.nn.tanh, lambda x: x]    
    n_train = 10
    x = np.random.normal(size=(n_train, 1))
    y = x * np.sin(x) * np.cos(x)

    V_fn = get_potential_energy_fn(x=x, y=y, forward_fn=lambda x, q: forward(x=x, weights=q, activations=activations))
    grad_V_fn = jax.jit(jax.grad(V_fn, argnums=(0)))

    chain = []
    num_results = 100
    start = time.perf_counter()

    for _ in trange(num_results):
        params = hmc_step(
            params=params,
            V_fn=V_fn,
            K_fn=kintetic_energy,
            num_leapfrog_steps=60,
            step_size=0.0001,
        )
        chain.append(params)
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds")



    print(f"params before = {params}")

    params = hmc_step(
        params=params,
        V_fn=V_fn,
        K_fn=kintetic_energy,
        num_leapfrog_steps=60,
        step_size=0.0001,
    )
    print(f"params after = {params}")

    

    # grad_fn = jax.grad(loss_fn, argnums=(2))
    # grads = grad_fn(x, y, weights, activations)
    # print(grads)
    # print(grads)
    
    # print(weights)
    # new_weights = jax.tree_multimap(
    #     update_fn,
    #     weights, 
    #     grads,
    # )
    print("--" * 30)
    # print(new_weights)



    # x = np.random.normal(size=1)
    # print(forward(x, weights, activations))
    
    # print(grad(x, weights, activations))

    # print(dense(x, w, b))

    

if __name__ == "__main__":
    main()