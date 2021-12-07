# Neural network codes

The neural network is written as a collection of classes. The main class is *FFNN* which is the neural network declared in [neural_network.hpp](https://github.com/reneaas/fys-stk4155/blob/master/project2/codes/neural_network/neural_network.hpp). The corresponding constructor and methods are found in [neural_network.cpp](https://github.com/reneaas/fys-stk4155/blob/master/project2/codes/neural_network/neural_network.cpp). In addition, a class called *Layer* is defined as a friend class of *FFNN* and keeps track of properties of each layer. For further code documentation, we advice you to take a look in the header files.


## Running test codes
To illustrate the usage of the codes, we've written a makefile that compiles and runs a few test runs.

Testing the code on a regression task can be achieved by running the following in a Linux/Unix command line:

```terminal
make regression
```

Testing the code on a classification task can be achieved by running the following in a Linux/Unix command line:

```terminal
make classification
```

## Usage of the neural network class
For regression tasks, the code *regression_main.cpp* illustrates its usage applied to data generated from Franke's function. For classification tasks, the code *classification_main.cpp* illustrates its use applied to the MNIST dataset.


## Required libraries
The C++ codes are created using the linear algebra and scientific computing library [Armadillo](http://arma.sourceforge.net/). Usage of these codes thus requires installation of this software which can be downloaded for free from http://arma.sourceforge.net/.
