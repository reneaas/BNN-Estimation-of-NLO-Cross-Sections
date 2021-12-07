import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

with tf.device("/CPU:0"):
    normals_2d = [
    tfd.MultivariateNormalDiag([0, 0], [1, 1]),
    tfd.MultivariateNormalDiag([4, 4], [1, 1]),
    ]
    bimodal_gauss = tfd.Mixture(tfd.Categorical([1, 1]), normals_2d)

    @tf.function
    def sample_chain(*args, **kwargs):
        return tfp.mcmc.sample_chain(*args, **kwargs)

    step_size = 1e-3
    # The HamiltonianMonteCarlo kernel works with both simple and dual averaging adaptation.
    # kernel = tfp.mcmc.HamiltonianMonteCarlo(
    #     bimodal_gauss.log_prob, step_size=step_size, num_leapfrog_steps=3
    # )
    # On the other hand, the NoUTurnSampler fails with both.
    kernel = tfp.mcmc.NoUTurnSampler(bimodal_gauss.log_prob, step_size=step_size)
    adaptation_steps = 100
    # adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    #     kernel, num_adaptation_steps=adaptation_steps
    # )
    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=adaptation_steps
    )

    chain = sample_chain(
        kernel=adaptive_kernel, num_results=100, current_state=tf.constant([0.0, 0.0]), trace_fn=None
    )

