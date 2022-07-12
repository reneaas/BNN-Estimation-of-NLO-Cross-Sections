# Bayesian Neural Networks Trained with Hamiltonian Monte Carlo

We provide three implementations:

1. [Object-oriented BNN in TensorFlow-Probability](./bnn/bnn.py)
2. [Functional BNN in TensorFlow-Probability](./bnn_functional/bnn.py)
3. [Functional BNN in Jax](./jax/jax_bnn.py)



This document will throughly describe the usage of the
bayesian neural network (BNN) trained with Hamiltonian Monte Carlo (HMC) and its derivatives.

## Basic bayesian learning with BNNs

The BNN implemented here supports training for regression tasks.

To initialize a network, we need to initialize an instance of
the underlying `BayesianNeuralNetwork` class as follows: 

```python
layers = [1, 50, 50, 1]
activations = ["swish", "relu", "identity"]
lamb = 1e-3
likelihood_noise = 0.1
bnn = BayesianNeuralNetwork(
    layers=layers,
    activations=activations,
    lamb=lamb,
    likelihood_noise=likelihood_noise,
)
```

This will initialize parameters from a default prior which is Gaussian, though a specific prior can also be specific if desirable.

## Training the BNN

The typical training of a BNN in the bayesian framework 
can be summarized as

1. Perform backpropagation for a number of `num_epochs` with a given
`batch_size` to reach a high probability region of the target probability distribution (the posterior).

2. Perform Monte Carlo sampling using HMC or its derivatives with a small number of burn-in steps to adapt step size.

This can be done with the following code example:

```python

# Perform backpropagation to reach a high probability region of the posterior.
num_epochs = 10000
batch_size = 32
bnn.mle_fit(
    x_train=x_train,
    y_train=y_train,
    num_epochs=num_epochs,
    batch_size=batch_size,
)

# Sample network weights from the posterior.
num_results = 1000
num_burnin_steps = 100
num_steps_between_results = 10 #Skip some samples to reduce correlation
step_size = 0.001

inner_kernel = tfp.mcmc.NoUTurnSampler(
    bnn.target_log_prob_fn(x=x_train, y=y_train)
    step_size = [step_size for _ in bnn.weights]
)
kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=inner_kernel,
    num_adaptation_steps=int(0.8 * num_burnin_steps),
)

chain = bnn.sample_chain(
    kernel=kernel,
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    num_steps_between_results=num_steps_between_results,
    current_state=bnn.weights,
    trace_fn=None,
)
```