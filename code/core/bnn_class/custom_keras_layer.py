import tensorflow as tf
import tensorflow_probability as tfp


class BNN(tf.keras.Sequential):
    """Wrapper for Sequential model that modifies the `set_weights` method to
    allow for change of parameters in the computational graph.
    """

    def __init__(self, layers=None):
        super(BNN, self).__init__(layers=layers)

        self._var = 2
    
    def set_weights(self, weights):
        kernel = weights[::2]
        bias = weights[1::2]
        for layer, w, b in zip(self.layers, kernel, bias):
            layer._weights = [w, b]
            layer._kernel = w
            layer._bias = b
    
    @property
    def var(self):
        return self._var 


@tf.function
def set_new_weights(new_weights):
    model.set_weights(new_weights)


if __name__ == "__main__":
    #model = BNN()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=50, input_shape=(1,)))
    print(model.summary())

    weights = model.get_weights()
    print(weights)

    new_weights = [tf.random.normal(w.shape) for w in weights]
    set_new_weights(new_weights)
    #model.set_weights(new_weights)
    print(model.get_weights())

