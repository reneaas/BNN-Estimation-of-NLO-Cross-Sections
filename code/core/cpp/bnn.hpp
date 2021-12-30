#ifndef BNN_HPP
#define BNN_HPP

#include <layer.hpp>
#include <vector>
#include <armadillo>

class BNN {
private:
    /* data */
    //std::vector<Layer> layers_;
    //std::vector<BNN::*> activations_;


public:
    BNN(){}; //Empty constructor.
    BNN(std::vector<Layer> layers);

    int num_layers_;
    double learning_rate_;
    double l2_strength_;
    void add(int n_rows, int n_cols, std::string activation);
    arma::mat forward(arma::mat x);
    void backward(arma::mat x, arma::mat y);
    void apply_gradients();
    void mle_fit(
        arma::mat x,
        arma::mat y,
        int num_epochs,
        double learning_rate
    );

    double loss(arma::mat yhat, arma::mat y);

    std::vector<arma::mat> hmc_step(
        arma::mat x, 
        arma::mat y,
        int num_leapfrog_steps,
        int step_size
    );

    arma::mat compute_r2(arma::mat x, arma::mat y);

    virtual ~BNN (){};

    /* Activation functions */ 
    arma::mat sigmoid(arma::mat z);
    arma::mat sigmoid_derivative(arma::mat z);
    
    arma::mat relu(arma::mat z);
    arma::mat relu_derivative(arma::mat z);

    arma::mat identity(arma::mat z);
    arma::mat identity_derivative(arma::mat z);


    // arma::mat (BNN::*hidden_act)(arma::vec z);
    // arma::mat (BNN::*top_layer_act)(arma::vec a);
    // arma::mat (BNN::*hidden_act_derivative)(arma::vec z);


    // void (FFNN::*update_parameters)();




    std::vector<Layer> layers_;
    std::vector<arma::mat (BNN::*)(arma::mat)> acts_;
    std::vector<arma::mat (BNN::*)(arma::mat)> acts_derivative_;
};

#endif
