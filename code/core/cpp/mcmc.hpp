#ifndef MCMC_HPP
#define MCMC_HPP

#include "bnn.hpp"
#include "layer.hpp"
#include <armadillo>
//#include <omp.h>

class MCMC {
private:
    /* data */
    std::vector<arma::mat> samples_;
    double l2_strength_;

    std::vector<arma::mat> momentum_kernel_;
    std::vector<arma::mat> momentum_bias_;


    //Pointer to member variables
    //arma::vec (FFNN::*hidden_act_derivative)(arma::vec z);

public:
    MCMC();
    MCMC(double l2_strength);

    std::vector<arma::mat> hmc_step(
        BNN *model, arma::mat x, arma::mat y, int num_leapfrog_steps, double step_size
    );

    // std::vector<std::vector<arma::mat>> sample_chain_hmc(
    //     BNN &model, 
    //     arma::mat x, 
    //     arma::mat y, 
    //     int num_results, 
    //     int num_leapfrog_steps, 
    //     double step_size
    // );

    // double get_potential_energy(
    //     std::vector<Layer>, arma::mat yhat, arma::mat y
    // );
    double get_kinetic_energy(
        BNN model,
        std::vector<arma::mat> p_w, 
        std::vector<arma::mat> p_b
    );


};

#endif