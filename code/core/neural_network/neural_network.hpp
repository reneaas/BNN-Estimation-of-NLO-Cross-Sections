#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "layer.hpp"
#include <armadillo>
#include <random>

/*

The class FFNN is a feed forward neural network.
Parameters passed to the constructor:

int hidden_layers: Number of hidden layers in the model excluding the input layer (which is treated as an input not part of the model)
int features: Length of the input vector x
int nodes: number of nodes on each hidden layer
int outputs: number of nodes in the top layer (number of outputs)
string model_type: either "classification" or "regression" depending on the model task.
double lamb: L2 regularization parameter.
double gamma: SGD momentum parameter
string hidden_activation: specifies which hidden activation function to use. Either "sigmoid", "relu" or "leaky_relu".



If the model is passes model_type = "classification", it will default to a binary classifier if outputs = 1, otherwise it will
use a softmax classifier.
*/

class FFNN {
private:
    friend class Layer;

    std::vector<Layer> layers_;

    int hidden_layers_, features_, nodes_, num_outputs_, num_points_, num_layers_;
    int epochs_, batch_sz_;
    double eta_;
    arma::mat X_train_, y_train_;
    arma::mat X_test_, y_test_;
    int num_test_;
    double lamb_, gamma_;


    //These methods make up the backpropagation algorithm
    void feed_forward(arma::vec x);
    void backward_pass(arma::vec x, arma::vec y);
    void add_gradients(int l);

    //Pointers to member functions
    arma::vec (FFNN::*hidden_act)(arma::vec z);
    arma::vec (FFNN::*top_layer_act)(arma::vec a);
    arma::vec (FFNN::*hidden_act_derivative)(arma::vec z);
    double (FFNN::*compute_metric)();
    void (FFNN::*update_parameters)();


    //Various hidden layer activation functions
    arma::vec sigmoid(arma::vec z);
    arma::vec sigmoid_derivative(arma::vec z);

    arma::vec relu(arma::vec z);
    arma::vec relu_derivative(arma::vec z);

    arma::vec leaky_relu(arma::vec z);
    arma::vec leaky_relu_derivative(arma::vec z);

    //Top layer activation functions
    arma::vec softmax(arma::vec a);
    arma::vec linear(arma::vec z);
    arma::vec binary_classifier(arma::vec z);

    //Various metrics
    double compute_accuracy();
    double compute_r2();
    double compute_mse();


    //Various update rules
    void update();
    void update_l2();
    void update_l2_momentum();


public:
    //Constructors
    FFNN(int features, int outputs, std::string model_type, double lamb, double gamma, std::string hidden_activation);
    FFNN(int hidden_layers, int features, int nodes, int outputs, std::string model_type, double lamb, double gamma, std::string hidden_activation);

    //Member functions
    void add_layer(int rows, int cols); //Adds a layer to the model.
    void init_data(arma::mat X_train, arma::mat y_train); //Initializes training data.
    void fit(int epochs, int batch_sz, double eta); //Fits the model
    double evaluate(arma::mat X_test, arma::mat y_test); //Evaluates the model according to the appropriate performance metric.


};

#endif
