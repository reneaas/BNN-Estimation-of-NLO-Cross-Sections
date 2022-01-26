import jax.numpy as njp
from functools import partial
import jax
from typing import Callable
import numpy as np

def midpoint(f, points):
    integral = 0
    for elem in points:
        integral += f(elem)
    return integral

@jax.jit
def tryout(n):
    s = 0
    for i in range(n):
        s += i
    return s

def main():
    fn = lambda x: 1 / (1 + x ** 2)

    # integrator = partial(midpoint, fn=fn, static_argnums=(3,))
    # integrator = jax.tree_util.Partial(midpoint)
    # integrator = jax.jit(integrator)
    # integrator = jax.pmap(midpoint, in_axes=(None, 0))
    # x = np.linspace(start=0, stop=100, num=1000)
    # x = x[:, None]
    
    # integral = integrator(f=fn, points=x)
    # # integral = midpoint(a=0., b=100., n=1000)
    # print(f"{integral=}")

    print(tryout(n=5))

if __name__ == "__main__":
    main()