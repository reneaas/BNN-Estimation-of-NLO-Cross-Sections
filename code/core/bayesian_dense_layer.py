import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tf.random.set_seed(100)
np.random.seed(100)

tfd = tfp.distributions


class BayesianDenseLayer(tf.keras.layers.Dense):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(BayesianDenseLayer, self).__init__(
            units,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
        )

    def set_weights(self, kernel, bias):
        self.kernel = kernel
        self.bias = bias

    # def call(self, inputs):
    #     return self.activation(
    #         tf.einsum("ij,...j->...j", self.kernel, inputs) + self.bias
    #     )


@tf.function
def loss_fn(model, y, yhat, lamb=1e-5):
    loss = tf.keras.losses.MSE(y, yhat)
    tmp = 0
    for layer in model.layers:
        tmp += tf.nn.l2_loss(layer.kernel)
        tmp += tf.nn.l2_loss(layer.bias)
    loss += lamb * tmp
    return loss

@tf.function
def grad_loss(model, x, y):
    with tf.GradientTape() as tape:
        yhat = model(x)
        loss = loss_fn(model, y, yhat)
        loss = tf.reduce_mean(loss)

    grad = tape.gradient(loss, model.weights)
    return loss, grad


if __name__ == "__main__":
    units = 5
    n_train = 10
    x_train = tf.random.normal(shape=(n_train, 1))
    f = lambda x: tf.math.sin(x)
    y_train = f(x_train)
    model = tf.keras.Sequential(BayesianDenseLayer(units=units))
    yhat = model(x_train)
    kernel = tf.Variable(tf.random.normal(model.layers[0].kernel.shape))
    bias = tf.Variable(tf.random.normal(model.layers[0].bias.shape))
    print(model.trainable_variables)
    model.set_weights([kernel, bias])
    print(model.trainable_variables)
    # model.layers[0].kernel = kernel
    # model.layers[0].bias = bias
    # print(model.layers[0].kernel)
    # print("Trainiable paramters = \n")
    # print(model.trainable_variables)
    #print("--"*50, "\n")
    loss, grad = grad_loss(model, x_train, y_train)
    #print(grad)
