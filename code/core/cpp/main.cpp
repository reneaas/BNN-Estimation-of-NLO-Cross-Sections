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
        y(0, i) = x(0, i) * cos(x(0, i)) * sin(x(0, i));
    }
    return y;
}

using namespace std;
using namespace arma;

BNN get_model(){
    vector<Layer> layers;
    BNN bnn = BNN(layers);
    bnn.add(100, 1, "sigmoid");
    bnn.add(100, 100, "sigmoid");
    bnn.add(1, 100, "identity");
    return bnn;
}

int main(int argc, char const *argv[]) {

    int n_train = 10000;
    int dims = 1;
    arma::mat x_train = arma::mat(dims, n_train).randn() * 3;
    arma::mat y_train = f(x_train);

    BNN model = get_model();

    arma::mat yhat = model.forward(x_train);
    // // yhat.print("yhat = ");
    model.backward(x_train, y_train);
    std::cout << "Passed forward and backward pass" << std::endl;

    // int num_epochs = 1000;
    // double learning_rate = 0.01;
    model.mle_fit(
        x_train, y_train, 1000, 0.001
    );

    arma::mat r2 = model.compute_r2(x_train, y_train);
    cout << "r2 = " << r2 << endl;
    int num_results = 1000;
    int num_burnin_steps = 100;
    MCMC mcmc = MCMC();
    std::vector<arma::mat> new_weights;
    for (int i = 0; i < num_burnin_steps; i++) {
        new_weights = mcmc.hmc_step(
            model, x_train, y_train, 60, 1e-4
        );
    }
    // for (auto param : new_weights) {
    //     param.print("param = ");
    // }
    std::vector<arma::cube> weights;
    for (auto param : new_weights) {
        weights.push_back(
            arma::cube(param.n_rows, param.n_cols, num_results).fill(0.)
        );
    }

    for (int i = 0; i < num_results; i++) {
        std::cout << "i = " << i << "\n";
        new_weights = mcmc.hmc_step(
            model, x_train, y_train, 60, 1e-4
        );
        for (std::size_t j = 0; j < new_weights.size(); j++) {
            weights[j].slice(i) = new_weights[j];
        }
    }

    for (std::size_t i = 0; i < weights.size(); i++) {
        weights[i].save("models/weight:" + std::to_string(i));
    }










    return 0;
}
