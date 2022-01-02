import tensorflow as tf
import tensorflow_probability as tfp
import time 
import seaborn as sns
import matplotlib.pyplot as plt


def target_log_prob_fn(x: tf.Tensor) -> tf.Tensor:
    return -0.5 * tf.reduce_sum(x ** 2, axis=-1)

@tf.function
def sample_chain(*args, **kwargs):
    return tfp.mcmc.sample_chain(*args, **kwargs)

def main() -> None:
    num_chains = 250000
    num_dims = 1
    num_results = 100
    num_burnin_steps = 100
    step_size = 0.01
    num_leapfrog_steps = 20
    x = tf.random.normal(shape=(num_chains, num_dims), mean=0.0, stddev=3.0)


    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )

    # kernel = tfp.mcmc.RandomWalkMetropolis(
    #     target_log_prob_fn=target_log_prob_fn
    # )

    # kernel = tfp.mcmc.NoUTurnSampler(
    #     target_log_prob_fn=target_log_prob_fn,
    #     step_size=step_size,
    # )

    start = time.perf_counter()
    chain = sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=x,
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

