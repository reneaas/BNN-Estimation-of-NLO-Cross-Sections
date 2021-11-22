import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class† BayesianDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(BayesianDenseLayer, self).__init__()
        self.num_outputs = num_outputs

        weights_prior = tfd.Normal(loc=0., scale=1.)
        bias_prior = tfd.Normal(loc=0., scale=1.)


    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                        shape=[
                            int(input_shape[-1]),
                            self.num_outputs
                        ]
        )

    def call(self, inputs, w, b):
        return self.activation(self.matmul(w, inputs) + b)
