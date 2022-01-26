import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np


class _BNNBase(object):
    def __init__(
        self,
        layers=None,
        activation=None,
        kernel_prior=None,
        bias_prior=None,
        num_chains=1,
    ):

        self._num_chains = num_chains

        self.avail_activations = {
                "sigmoid": tf.nn.sigmoid,
                "relu": tf.nn.relu,
                "leaky_relu": tf.nn.leaky_relu,
                "tanh": tf.nn.tanh,
                "swish": tf.nn.swish, #Alternatively called `SiLU`
                "identity": tf.identity,
            }
        
        if activation is not None and layers is not None:
            self.num_layers = len(layers) - 1

            # Check if list with an activation per layer.
            if isinstance(activation, list):
                assert len(activation) == len(layers) - 1, ValueError(
                    f"""len(activation) = {len(activation)} does not match the number of layers."""
                )

                self.activation = []
                for act in activation:
                    assert isinstance(act, str) or callable(act), TypeError(
                        f"activation = {act} is not of type `str` or is not callable."
                    )

                    assert (
                        act in self.avail_activations
                        or act in self.avail_activations.values()
                    ), ValueError(
                        f"""activation = {act} is not a valid activation. 
                        Available activations:
                        {list(self.avail_activations.keys())}.
                        Or provide the tf.nn equivalent to these. e.g 
                        {list(self.avail_activations.values())}
                        """
                    )
                    if isinstance(act, str):
                        self.activation.append(self.avail_activations.get(act))
                    else:
                        self.activation.append(act)

            # Check if user provides a single activation for the "hidden layers".
            elif isinstance(activation, str):
                assert activation in self.avail_activations, ValueError(
                    f"""activation = {activation} is not a valid activation. 
                    Available activations:
                    {list(self.avail_activations.keys())}.
                    Or provide the tf.nn equivalent to these. e.g 
                    {list(self.avail_activations.values())}
                    """
                )

                self.activation = [
                    self.avail_activations.get(activation)
                    for _ in range(len(layers) - 2)
                ]
                self.activation.append(tf.identity)  # Activation for output layer.

            elif callable(activation):
                assert activation in self.avail_activations.values(), ValueError(
                    f"""activation = {activation} is not a valid activation. 
                    Available activations:
                    {list(self.avail_activations.keys())}.
                    Or provide the tf.nn equivalent to these. e.g 
                    {list(self.avail_activations.values())}
                    """
                )

                self.activation = [activation for _ in range(len(layers) - 2)]
                self.activation.append(tf.identity)  # Activation for output layer.
            # Set priors of kernel
            if kernel_prior is not None:
                self.kernel_prior = kernel_prior
            else:
                self.kernel_prior = tfp.distributions.Normal(loc=0.0, scale=1.0)

            # Set priors of bias
            if bias_prior is not None:
                self.bias_prior = bias_prior
            else:
                self.bias_prior = tfp.distributions.Normal(loc=0.0, scale=1.0)

            self.weights = self._create_layers(layers)
    
    def _create_layers(self, layers):
        """Helper function to create the weights per layer of the model.

            Args:
                layers (list):
                    List containing the number of nodes per layer.
                    Shape -> [input_sz, layer:0, layer:1, ..., output_sz].
        """
        weights = []
        for n, m in zip(layers[:-1], layers[1:]):
            weights.extend(
                [
                    self.kernel_prior.sample(sample_shape=(self._num_chains, n, m)),
                    self.bias_prior.sample(sample_shape=(self._num_chains, m)),
                ]
            )
        return weights
    
    def __str__(self):
        """Returns string with a summary of the model."""
        kernel = self.weights[::2]
        bias = self.weights[1::2]
        d = {
            "layer": [],
            "params_per_layer": [],
            "kernel_shape": [],
            "bias_shape": [],
            "activation": [],
        }
        tot_num_params = 0
        for i, (w, b, activation) in enumerate(zip(kernel, bias, self.activation)):
            num_params = (tf.reduce_prod(w.shape[1:]) + tf.reduce_prod(b.shape[1:])).numpy()
            tot_num_params += num_params
            d["layer"].append(i)
            d["params_per_layer"].append(num_params)
            d["kernel_shape"].append(str(w.shape))
            d["bias_shape"].append(str(b.shape))
            d["activation"].append(activation.__name__)

        s = "--" * 60 + "\n"
        s += "Model summary: \n"
        s += "--" * 60 + "\n"
        df = pd.DataFrame(data=d)
        s += df.to_string(index=False, col_space=20)
        s += "\n" + "--" * 60 + "\n"
        s += f"Number of parameters: {tot_num_params}\n"
        s += f"Number of sampled networks: {kernel[0].shape[0]}\n"
        s += "--" * 60 + "\n"
        return s
    
    def summary(self):
        """Alias for __str__."""
        return str(self)
    
    def save_model(self, fname, compressed=False, allow_pickle=False, chain=None):
        """Saves the weights and activations of the model to a numpy zip file (.npz)
        using numpy.savez or numpy.savez_compressed.
        By default it uses numpy.savez and avoids use of pickle to ensure
        port compatibility.

        Args:
            fname (str):
                Filename with ending ".npz".
            compressed (bool):
                Allow for compressed save option.
            allow_pickle (bool):
                Allow Python pickles or not.

        Raises:
            ValueError if `fname` does not end with `.npz`.
        """
        assert isinstance(fname, str), TypeError(f"`fname` = {fname} is not of type `str`.")
        if fname.endswith(".npz") is False:
            raise ValueError(
                f"""The filename does not end with .npz. Please choose a filename with .npz ending."""
            )
        if chain is None:
            kernel = self.weights[::2]
            bias = self.weights[1::2]
        else:
            kernel = chain[::2]
            bias = chain[1::2]
        
        weights = {}
        for i, (w, b, activation) in enumerate(zip(kernel, bias, self.activation)):
            weights[f"kernel:{i}"] = w.numpy()
            weights[f"bias:{i}"] = b.numpy()
            if isinstance(activation, str):
                weights[f"activation:{i}"] = activation
            elif callable(activation):
                weights[f"activations:{i}"] = activation.__name__
                # for key, val in self.avail_activations.items():
                #     if activation is val:
                #         weights[f"activation:{i}"] = key
        if compressed:
            np.savez_compressed(file=fname, **weights, allow_pickle=allow_pickle)
        else:
            np.savez(file=fname, **weights, allow_pickle=allow_pickle)
    
    def load_model(self, fname):
        """Loads a model from a filename `fname`.
        Loads the weights from a MCMC chain and activations of the model.

        Args:
            fname (str): Filename with the saved model.
        """
        data = np.load(fname)
        kernel = data.files[::3]
        bias = data.files[1::3]
        activation = data.files[2::3]

        self.weights = []
        self.activation = []
        for kernel_name, bias_name, activation_name in zip(kernel, bias, activation):
            self.weights.extend(
                [
                    tf.convert_to_tensor(data[kernel_name], name=kernel_name),
                    tf.convert_to_tensor(data[bias_name], name=bias_name),
                ]
            )
            self.activation.append(
                self.avail_activations.get(str(data[activation_name]))
            )




def main():
    layers = [1, 50, 1]
    activation = ["swish", "identity"]
    bnn_base = _BNNBase(layers=layers, activation=activation)
    print(bnn_base)

if __name__ == "__main__":
    main()