import tensorflow as tf
import tensorflow_probability as tfp
import time
import seaborn as sns
import matplotlib.pyplot as plt
import sys



def sample_chain(*args, **kwargs):
    return tfp.mcmc.sample_chain(*args, **kwargs)


def log_prior(current_state, lamb=1e-3):
    kernel = current_state[::2]
    bias = current_state[1::2]
    res = 0
    for w, b in zip(kernel, bias):
        res += tf.reduce_sum(w ** 2)
        res += tf.reduce_sum(b ** 2)
    return -0.5 * lamb * res

def likelihood_fn(x, y, current_state):
    kernel = current_state[::2]
    bias = current_state[1::2]
    for w, b in zip(kernel[:-1], bias[:-1]):
        w = tf.RaggedTensor.to_tensor(w)
        b = tf.RaggedTensor.to_tensor(b)
        x = tf.nn.tanh(
            tf.matmul(w, x)
        )
    w = tf.RaggedTensor.to_tensor(kernel[-1])
    b = tf.RaggedTensor.to_tensor(bias[-1])
    y_pred = tf.identity(
        tf.matmul(w, x)
    )


    # for w, b in zip(kernel[:-1], bias[:-1]):
    #     x = tf.nn.tanh(tf.linalg.matmul(x, w) + b)
    # y_pred = tf.linalg.matmul(x, kernel[-1]) + bias[-1]
    return -0.5 * tf.reduce_sum((y - y_pred) ** 2)
    
def get_target_log_prob_fn(x, y):
    def target_log_prob_fn(*current_state):
        probs = []
        for i, state in enumerate(current_state):
            probs.append(
                log_prior(state) + likelihood_fn(x, y, state)
            )
        return tf.convert_to_tensor(probs)
    return target_log_prob_fn

def get_init_state(layers, num_chains):
    init_state = []
    for _ in range(num_chains):
        state = []
        for n, m in zip(layers[:-1], layers[1:]):
            state.append(
                tf.random.normal(shape=(n, m))
            )
            state.append(
                tf.random.normal(shape=(1, m))
            )
        state = tf.ragged.stack(state)
        init_state.append(state)
    return init_state


def main() -> None:
    num_chains = 2
    num_dims = 1
    num_results = 100
    num_burnin_steps = 300
    step_size = 0.1
    num_leapfrog_steps = 60
    x = tf.random.normal(shape=(100, 1), mean=0.0, stddev=3.0)
    y = x * tf.math.sin(x) * tf.math.cos(x)

    layers = [1, 10, 1]
    current_state = get_init_state(layers, num_chains)
    # print(current_state)
    for state in current_state:
        print(likelihood_fn(x, y, state))
    # target_log_prob_fn = get_target_log_prob_fn(x, y)
    # log_prob = target_log_prob_fn(*current_state)
    # print(log_prob)
    sys.exit()

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=get_target_log_prob_fn(x, y),
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )

    # kernel = tfp.mcmc.NoUTurnSampler(
    #     target_log_prob_fn=target_log_prob_fn, step_size=step_size
    # )

    # kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    #     inner_kernel=kernel,
    #     num_adaptation_steps=int(0.8 * num_burnin_steps),
    #     target_accept_prob=0.75,
    # )


    start = time.perf_counter()
    chain = sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=current_state,
        trace_fn=None
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"{timeused=} seconds")

    chain = chain.numpy().reshape(-1, num_dims)
    print(chain.shape)

    sns.histplot(chain.ravel())
    plt.show()


if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()
