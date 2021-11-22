import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import seaborn as sns


def prior(q: np.ndarray) -> float:
    return 0.5 * np.dot(q, q)


def grad_prior(q: np.ndarray) -> np.ndarray:
    return np.array([1, *q])


def model(x: np.ndarray, q: np.ndarray) -> float:
    """Linear regression model

        Args:
            q : model params
            x : input point
    """
    return np.dot(x, q)

def likelihood(x: np.ndarray, q: np.ndarray, t: float, model: float) -> float:
    y = model(x, q)
    return (y - t) ** 2


def V(q: np.ndarray, x: np.ndarray, t: np.ndarray) -> float:
    return 0.5*np.dot(q, q) + np.sum(model(x,q)-t)


def grad_V(q: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return q + np.einsum("i,ij->j", model(x,q)-t, x)


def hmc(
    V: float,
    grad_V: np.ndarray,
    eps: float,
    L: int,
    current_q: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Performs one step of Hamiltonian Monte Carlo

    Args:
        V (function)        :   Potential energy as a function of params q
        grad_V (function)   :   Gradient of potential energy
        eps                 :   Step size in Leapfrog method
        L                   :   Number of Leapfrog iterations
        current_q           :   Generalized coordinate (or parameter).
    """

    q = np.copy(current_q)
    q = current_q
    p = np.random.normal(0, 1, size=q.shape)
    current_p = np.copy(p)
    # Make a half step for momentum at the beginning:
    p = p - 0.5 * eps * grad_V(q, x, t)

    # Do the inner steps of Leapfrog
    for i in range(L - 1):
        q += eps * p
        p -= eps * grad_V(q, x, t)

    # Final step of Leapfrog
    q += eps * p
    p -= 0.5 * eps * grad_V(q, x, t)
    p = -p

    current_V = V(current_q, x, t)
    current_K = 0.5 * np.dot(current_p, current_p)
    proposed_V = V(q, x, t)
    proposed_K = 0.5 * np.dot(p, p)

    dV = proposed_V - current_V
    dK = proposed_K - current_K

    if np.random.uniform() >= min(1, np.exp(-dV) * np.exp(-dK)):
        current_q[:] = q[:]
    return current_q

def ground_truth(x) -> np.ndarray:
    w = np.array([-2, 1])
    return np.einsum("ij,j", x, w)


if __name__ == "__main__":
    n_train = 10000
    x_train = np.random.uniform(0, 1, size=(n_train, 2))
    x_train[:, 0] = 1
    t = ground_truth(x_train)
    q = np.random.normal(0, 1, size=2)
    num_samples = 100
    params = []
    predictions = []
    for k in range(num_samples):
        q = hmc(V=V, grad_V=grad_V, eps=0.01, L=40, x=x_train, t=t, current_q=q)
        params.append(np.copy(q))
    n_test = 1000
    x_test = np.random.uniform(0, 1, size=(n_test, 2))
    x_test[:, 0] = 1
    for i in range(len(params)):
        tmp = []
        for j in range(x_test.shape[0]):
            y = model(x=x_test[j], q=params[i])
            tmp.append(y)
            predictions.append(y)
    predictions = np.array(predictions)
    X = np.array(list(x_test[:, 1])*num_samples)
    sns.lineplot(x=X, y=predictions[:])
    plt.plot(x_test[:, 1], ground_truth(x_test), label="ground truth")
    plt.legend()
    plt.show()
