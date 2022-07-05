import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import numpy as np


class _BNNBase(object):
    """Internal Base class for Bayesian neural networks
    trained with Bayesian MCMC inference algorithms wrapped around
    TensorFlow Probability's functionality.
    """
    def __init__(
        self,
        layers=None,
        activations=None,
        kernel_prior=None,
        bias_prior=None,
        num_chains=1,
    ):

        self._num_chains = num_chains

        self._avail_activations = {
                tf.nn.sigmoid.__name__: tf.nn.sigmoid,
                tf.nn.relu.__name__: tf.nn.relu,
                tf.nn.leaky_relu.__name__: tf.nn.leaky_relu,
                tf.nn.tanh.__name__: tf.nn.tanh,
                tf.nn.swish.__name__: tf.nn.swish,
                tf.identity.__name__: tf.identity,
                tf.nn.elu.__name__: tf.nn.elu,
                tf.nn.gelu.__name__: tf.nn.gelu,
            }
        
        if activations is not None and layers is not None:
            self._num_layers = len(layers) - 1

            # Check if list with an activation per layer.
            if isinstance(activations, list):
                assert len(activations) == len(layers) - 1, ValueError(
                    f"""len(activations) = {len(activations)} does not match the number of layers."""
                )

                self._activations = []
                for act in activations:
                    assert isinstance(act, str) or callable(act), TypeError(
                        f"activation function = {act} is not of type `str` or is not callable."
                    )

                    assert (
                        act in self._avail_activations
                        or act in self._avail_activations.values()
                    ), ValueError(
                        f"""activation function = {act} is not a valid activation. 
                        Available activations:
                        {list(self._avail_activations.keys())}.
                        """
                    )
                    if isinstance(act, str):
                        self._activations.append(self._avail_activations.get(act))
                    else:
                        self._activations.append(act)

            # Check if user provides a single activation for the "hidden layers".
            elif isinstance(activations, str):
                assert activations in self._avail_activations, ValueError(
                    f"""activation function = {activations} is not a valid activation. 
                    Available activations:
                    {list(self._avail_activations.keys())}.
                    """
                )

                self._activations = [
                    self._avail_activations.get(activations)
                    for _ in range(len(layers) - 2)
                ]
                self._activations.append(tf.identity)  # Activation for output layer.

            elif callable(activations):
                assert activations in self._avail_activations.values(), ValueError(
                    f"""activation = {activations} is not a valid activation. 
                    Available activations:
                    {list(self._avail_activations.keys())}.
                    """
                )

                self._activations = [activations for _ in range(len(layers) - 2)]
                self._activations.append(tf.identity)  # Activation for output layer.
            # Set priors of kernel
            if kernel_prior is not None:
                self._kernel_prior = kernel_prior
            else:
                self._kernel_prior = tfp.distributions.Normal(loc=0.0, scale=1.0)

            # Set priors of bias
            if bias_prior is not None:
                self._bias_prior = bias_prior
            else:
                self._bias_prior = tfp.distributions.Normal(loc=0.0, scale=1.0)

            self._weights = self._create_layers(layers)

    @property
    def num_chains(self):
        return self._num_chains
    
    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        self._weights = new_weights

    @property
    def num_layers(self):
        return self._num_layers
    
    @property
    def activations(self):
        return self._activations
    
    @property
    def avail_activations(self):
        return list(self._avail_activations.keys())

    @property
    def num_params(self):
        tot_num_params = 0
        for w, b in zip(self.weights[::2], self.weights[1::2]):
            num_params = (tf.reduce_prod(w.shape[1:]) + tf.reduce_prod(b.shape[1:])).numpy()
            tot_num_params += num_params
        return tot_num_params
    
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
                    self._kernel_prior.sample(sample_shape=(self._num_chains, n, m)),
                    self._bias_prior.sample(sample_shape=(self._num_chains, m)),
                ]
            )
        return weights
    
    def __str__(self):
        """Returns string with a summary of the model."""
        kernel = self._weights[::2]
        bias = self._weights[1::2]
        d = {
            "layer": [],
            "params_per_layer": [],
            "kernel_shape": [],
            "bias_shape": [],
            "activation": [],
        }
        tot_num_params = 0
        for i, (w, b, activation) in enumerate(zip(kernel, bias, self._activations)):
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
            kernel = self._weights[::2]
            bias = self._weights[1::2]
        else:
            kernel = chain[::2]
            bias = chain[1::2]
        
        model = {}
        for i, (w, b, activation) in enumerate(zip(kernel, bias, self._activations)):
            model[f"kernel:{i}"] = w.numpy()
            model[f"bias:{i}"] = b.numpy()
            if isinstance(activation, str):
                model[f"activation:{i}"] = activation
            elif callable(activation):
                model[f"activations:{i}"] = activation.__name__
        if compressed:
            np.savez_compressed(file=fname, **model, allow_pickle=allow_pickle)
        else:
            np.savez(file=fname, **model, allow_pickle=allow_pickle)
    
    def load_model(self, fname):
        """Loads a model from a filename `fname`.
        Loads the weights from a MCMC chain and activations of the model.

        Args:
            fname (str): Filename with the saved model.
        """
        model = np.load(fname)
        kernel = model.files[::3]
        bias = model.files[1::3]
        activations = model.files[2::3]

        self._weights = []
        self._activations = []
        for kernel_name, bias_name, activation_name in zip(kernel, bias, activations):
            self._weights.extend(
                [
                    tf.convert_to_tensor(model[kernel_name], name=kernel_name, dtype=tf.float32),
                    tf.convert_to_tensor(model[bias_name], name=bias_name, dtype=tf.float32),
                ]
            )
            self._activations.append(
                self._avail_activations.get(str(model[activation_name]))
            )

    @tf.function
    def _sample_chain(self, *args, **kwargs):
        """A simple wrapper around tfp.mcmc.sample_chain that speeds up code
        by compiling to a computational graph using tf.function.
        """
        return tfp.mcmc.sample_chain(*args, **kwargs)

    def _get_dataset(self, x, y, batch_size=None):
        """Creates a tf.data.Dataset object from training data.

        Args:
            x (tf.Tensor):
                Training features of shape (num_train, num_features)
            y (tf.Tensor):
                Training targets of shape (num_train, num_outputs)
            batch_size (optional, int):
                Batch size of dataset. Default: batch_size=16.

        Returns:
            ds (tf.data.Dataset):
                Dataset split into batches of size `batch_size`.
        """
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if batch_size is not None:
            ds = ds.batch(batch_size=batch_size)
        return ds

    @tf.function
    def _dense_layer(self, x, w, b, activation):
        """Computes the output of a dense layer.

        Args:
            x (tf.Tensor):
                Input features of shape (num_points, num_features)
            w (tf.Tensor):
                Kernel of layer. Shape: (batch_size, n, m)
            b (tf.Tensor):
                Bias of layer. Shape: (batch_size, m, )
            activation (callable):
                Activation function. Python callable.

        Returns:
            Computes activations of shape (batch_size, num_points, num_outputs)
        """
        return activation(tf.matmul(x, w) + b[..., None, :])
    
    def _restack_chain(self, chain):
        """Rearranges the shape of the chain.

        Args:
            chain (list[tf.Tensor]): List of sampled weights.

        Returns:
            new_chain (list[tf.Tensor]): New rearranged chain.
        """
        new_chain = []
        for w, b in zip(chain[::2], chain[1::2]):
            new_chain.extend(
                [
                    tf.reshape(
                        tensor=w,
                        shape=(w.shape[0] * w.shape[1], w.shape[2], w.shape[3]),
                    ),
                    tf.reshape(tensor=b, shape=(b.shape[0] * b.shape[1], b.shape[2])),
                ]
            )
        return new_chain




def main():
    layers = [1, 50, 1]
    activations = ["swish", "identity"]
    bnn_base = _BNNBase(layers=layers, activations=activations)

if __name__ == "__main__":
    main()