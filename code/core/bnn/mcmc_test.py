import tensorflow as tf
import tensorflow_probability as tfp
import time
import seaborn as sns
import matplotlib.pyplot as plt
import sys




@tf.function
def target_log_prob_fn(x):
    return -0.5 * x ** 2

@tf.function
def sample_chain(*args, **kwargs):
    kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.5
    )
    return tfp.mcmc.sample_chain(kernel=kernel, *args, **kwargs)


def main():
    # kernel = tfp.mcmc.NoUTurnSampler(
    #     target_log_prob_fn=target_log_prob_fn,
    #     step_size=0.5
    # )

    start = time.perf_counter()
    chain = sample_chain(
        num_results=int(2 ** 10),
        num_burnin_steps=10_000,
        trace_fn=None,
        current_state=tf.zeros([2 ** 10])
    )
    end = time.perf_counter()
    timeused = end - start
    print(f"timeused = {timeused} seconds")



if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()
