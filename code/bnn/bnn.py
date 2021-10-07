import tensorflow as tf
import tensorflow_probability as tfp

tfd = tf.distributions
# tf.enable_v2_behaviour()

def prior(kernel_size, bias_size, dtype = None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc = tf.zeros(n),
                    scale_diag = tf.ones(n)
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

def get_model(num_samples, num_dims):
    kl_divergence_fn = (lambda q, p, _: tfd.kl_divergence(q, p) \
                        / tf.cast(num_samples, dtype = tf.float32)
    )

    input_layer = tf.keras.layers.Input(shape = num_dims)
    dense_layer = tfp.layers.DenseFlipout(

    )



    model = tf.keras.models.Sequential(
            tfp.layers()
    )
