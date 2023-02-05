import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bnn.bnn import BayesianNeuralNetwork
import matplotlib.pyplot as plt
import seaborn as sns




def main(
    num_points=1000,
    num_results=1024,
    num_burnin_steps=1024,
    num_leapfrog_steps=128,
    step_size=1e-3,
):
    # Create a mock dataset 
    dims = 1
    x_train = np.random.normal(loc=0., scale=1., size=(num_points, dims)) # features 
    f = lambda x:  x * np.sin(x) * np.cos(x)  # Ground truth of the mock dataset
    y_train = f(x_train) + np.random.normal(loc=0., scale=0.1, size=(num_points,1)) # Targets with added noise

    # Convert to TensorFlow tensors with dtype tf.float32 (necessary step)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    
    # Setup the Bayesian neural network
    bnn = BayesianNeuralNetwork(
        layers=[dims, 10, 10, 1],
        activations="tanh",
        lamb=1e-3, # Regularization strength of the prior
        likelihood_noise=1., # "Regularization" of the likelihood
        num_chains=1, # Run a single independent Markov chain 
    )

    # Pretrain the BNN model. Returns an array of loss calculated per epoch of pretraining. 
    loss = bnn.mle_fit(
        x_train=x_train,
        y_train=y_train,
        epochs=1024,
        batch_size=32,
        lr=1e-3, 
    )

    # Set up samplers to setup Markov chain and infer neural network weights
    # All documentation for these are at https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc

    # Define some basic parameters
    num_results = 100 # Sample 1000 neural networks
    step_size = 0.001 # Initial step size
    num_burnin_steps = 1000 # Warm-up steps. Some of these will be used for adaptation, rest for burn-in.
    num_leapfrog_steps = 512 # Number of Leapfrog steps integrated in HMC.
    target_log_prob_fn = bnn.get_target_log_prob_fn(x=x_train, y=y_train) # Setup target log probability function.
    
    # First set up the fundamental transition kernel, basic HMC in this case
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=step_size,
        target_log_prob_fn=target_log_prob_fn,
    )
    
    # Set up step size adaptation
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps),
        target_accept_prob=0.75,
    )

    # Sample Markov chain of neural network models. After this, the model is "trained".
    chain = bnn.sample_chain(
        kernel=kernel,
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        num_steps_between_results=0,
        trace_fn=None,
        fname=None, # If fname is set to none, the model is not saved. Provide path with automatically save model. 
    )
    print([w.shape for w in chain])

    # Compute predictive distributions
    x = np.linspace(-1, 1, 1001)
    x = x[:, None] # shape (1001, 1)
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    y_pred = bnn(x).numpy().squeeze(-1) # Empirical predictive distribution of each target `y`` for each input point `x`.
    y_mean = np.mean(y_pred, axis=0) # Sample mean of each predictive distribution.
    y_std = np.std(y_pred, axis=0) # Sample standard deviation of each predictive distribution.

    # Plotting the resulting predictions 
    x = x.numpy().squeeze(-1) # Convert back to numpy array for and squeeze redundant dimension (num_points, 1) --> (num_points,)
    plt.plot(x, y_mean, color="blue", label="Mean predictions")
    plt.fill_between(
        x,
        (y_mean - y_std),
        y_mean + y_std,
        alpha=0.5,
    )
    plt.plot(x, f(x), label="Ground truth", color="red")    
    plt.legend()

    plt.show()
    print(f"{y_pred.shape = }")




if __name__ == "__main__":
    with tf.device("/CPU:0"):
        main()