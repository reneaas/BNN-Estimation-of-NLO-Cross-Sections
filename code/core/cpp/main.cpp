#include "layer.hpp"
#include "bnn.hpp"
#include "mcmc.hpp"
#include <armadillo>
#include <iostream>
#include <vector>
#include <random>
#include <time.h>

arma::mat sigmoid(arma::mat z) {
    return 1./(1. + exp(-z));
}

arma::mat f(arma::mat x) {
    arma::mat y = arma::mat(1, x.n_cols);
    for (int i = 0; i < (int) x.n_cols; i++){
        y(0, i) = sin(x(0, i));
    }
    return y;
}

using namespace std;
using namespace arma;

BNN get_model(){
    vector<Layer> layers;
    BNN bnn = BNN(layers);
    bnn.add(50, 1, "relu");
    // bnn.add(50, 20, "sigmoid");
    bnn.add(1, 50, "identity");
    return bnn;
}

int main(int argc, char const *argv[]) {

    int n_train = 10000;
    int dims = 1;
    arma::mat x_train = arma::mat(dims, n_train).randn();
    arma::mat y_train = f(x_train);

    BNN model = get_model();

    // arma::mat yhat = bnn.forward(x_train);
    // //yhat.print("yhat = ");
    // bnn.backward(x_train, y_train);

    // int num_epochs = 1000;
    // double learning_rate = 0.01;

    // bnn.mle_fit(x_train, y_train, num_epochs, learning_rate);
    // arma::mat r2 = bnn.compute_r2(x_train, y_train);
    // cout << "r2 = " << r2 << endl;
    MCMC mcmc = MCMC();
    std::vector<std::vector<arma::mat>> res;
    model.layers_[0].w_.print("w0 before = ");
    res = mcmc.hmc_step(model, x_train, y_train, 60, 0.001);
    
    std::cout << "res size = " << res.size() << std::endl;
    res[0][0].print("w0 after = ");
    return 0;
}

