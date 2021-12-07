#include "layer.hpp"
#include "neural_network.hpp"
#include <time.h>
#include <iostream>
#include <cstdio>
#include <fstream>

using namespace std;
using namespace arma;



void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test);
void test_mnist(int hidden_layers, int nodes, double lamb, double gamma, int epochs, int batch_sz, double eta, string hidden_act);

int main(int argc, char const *argv[]) {

    int hidden_layers = 1;
    int nodes = 10;
    double lamb = 1e-5;
    double gamma = 0.5;
    int epochs = 10;
    int batch_sz = 10;
    double eta = 0.01;
    string hidden_act = "relu";
    test_mnist(hidden_layers, nodes, lamb, gamma, epochs, batch_sz, eta, hidden_act);
    return 0;
}

void test_mnist(int hidden_layers, int nodes, double lamb, double gamma, int epochs, int batch_sz, double eta, string hidden_act)
{
    int features = 28*28;
    int num_outputs = 10;
    string model_type = "classification";

    int num_train = 0.95*60000;
    mat X_train = mat(features, num_train);
    mat y_train = mat(num_outputs, num_train);

    int num_val = 0.05*60000;
    mat X_val = mat(features, num_val);
    mat y_val = mat(num_outputs, num_val);

    int num_test = 10000;
    mat X_test = mat(features, num_test);
    mat y_test = mat(num_outputs, num_test);
    read_mnist(&X_train, &y_train, &X_val, &y_val, &X_test, &y_test);

    //Here we show how to use the neural net with several hidden layers with a variable number of nodes in the hidden layers in the context of classification.
    FFNN my_network(features, num_outputs, model_type, lamb, gamma, hidden_act);
    my_network.add_layer(nodes, features); //Add the first hidden layer connected to the input x.
    my_network.add_layer(2*nodes, nodes); //Add a hidden layer
    my_network.add_layer(3*nodes, 2*nodes); //Add another hidden layer
    my_network.add_layer(num_outputs, 3*nodes); //add top layer that produces the final prediction
    my_network.init_data(X_train, y_train); //Feed the training data to the model
    my_network.fit(epochs, batch_sz, eta); //Fit the model to the training data.
    double accuracy_val = my_network.evaluate(X_val, y_val); //Evaluate the model on the validation set
    double accuracy_test = my_network.evaluate(X_test, y_test); //Evaluate the model on the test set.

    cout << "validation accuracy = " << accuracy_val << endl;
    cout << "test accuracy = " << accuracy_test << endl;


    //Here we show a different version where every hidden layer has the same number of hidden neurons.
    //It calls a different constructor that sets up the model for us with no fuzz.
    FFNN my_network2(hidden_layers, features, nodes, num_outputs, model_type, lamb, gamma, hidden_act); //Create the neural net with all its layers.
    my_network2.init_data(X_train, y_train); //Feed the training data to the model
    my_network2.fit(epochs, batch_sz, eta); //Fit the model to the training data.
    accuracy_val = my_network2.evaluate(X_val, y_val); //Evalute the model on the validation set
    accuracy_test = my_network2.evaluate(X_test, y_test); //Evalute the model on the test set.

    cout << "validation accuracy = " << accuracy_val << endl;
    cout << "test accuracy = " << accuracy_test << endl;

}

void read_mnist(mat *X_train, mat *y_train, mat *X_val, mat *y_val, mat *X_test, mat *y_test){

    (*X_train).load("../datasets/mnist_X_train.bin");
    (*y_train).load("../datasets/mnist_y_train.bin");

    (*X_val).load("../datasets/mnist_X_val.bin");
    (*y_val).load("../datasets/mnist_y_val.bin");


    (*X_test).load("../datasets/mnist_X_test.bin");
    (*y_test).load("../datasets/mnist_y_test.bin");
}
