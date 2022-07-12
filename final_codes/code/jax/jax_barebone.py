import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import sys

# @partial(jax.jit, static_argnames=("weights", "activations"))
def forward(x, weights, activations):
    for w, b, act in zip(weights[::2], weights[1::2], activations):
        x = dense(x, w, b, act)
    return x

@partial(jax.jit, static_argnames=("activation"))
def dense(x, w, b, activation=jax.nn.relu):
    return activation(jnp.dot(x, w) + b)
# dense = jax.vmap(dense, in_axes=(0, None, None, None))

def loss_fn(x, y, weights, activations):
    x = jax.tree_multimap(lambda w: forward(weights=w, x=x, activations=activations), weights)
    return jnp.mean((x - y) ** 2)





def get_weights(layers):
    weights = []
    for n, m in zip(layers[1:], layers[:-1]):
        w = np.random.normal(size=(m, n)) / np.sqrt(m)
        b = np.random.normal(size=(n,))
        weights.extend([w, b])
    return weights
    

def update_fn(param, grad, lr=0.001):
    return param - grad * lr



def main():

    layers = [1, 10, 10, 1]
    weights = get_weights(layers=layers)
    activations = [jax.nn.relu, lambda x: x]    
    n_train = 16
    x = np.random.normal(size=(n_train, 1))
    y = x * np.sin(x) * np.cos(x)
    print(forward(x, weights, activations))
    sys.exit()

    grad_fn = jax.grad(loss_fn, argnums=(2))
    grads = grad_fn(x, y, weights, activations)
    print(grads)
    # print(grads)
    
    print(weights)
    new_weights = jax.tree_multimap(
        update_fn,
        weights, 
        grads,
    )
    print("--" * 30)
    print(new_weights)



    # x = np.random.normal(size=1)
    # print(forward(x, weights, activations))
    
    # print(grad(x, weights, activations))

    # print(dense(x, w, b))

    

if __name__ == "__main__":
    main()