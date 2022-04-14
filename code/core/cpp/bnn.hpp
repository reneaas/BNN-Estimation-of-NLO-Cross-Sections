#ifndef BNN_HPP
#define BNN_HPP

#include "layer.hpp"
#include <vector>
#include <random>

class BNN {
private:
    // RNG used in HMC.
    std::mt19937_64 gen_;
    std::normal_distribution<double> dist_;
    std::uniform_real_distribution<double> uniform_real_dist_;
    // std::uniform_int_distribution<int> uniform_int_dist_;

public:
    BNN(){}; //Nullary constructor
    BNN(std::vector<int> layers, std::string activation);
    virtual ~BNN(){};

    std::vector<Layer> layers_;
    std::vector<std::vector<double>> get_weights();
    void set_weights(std::vector<std::vector<double>> weights);
    std::vector<std::vector<double>> get_gradients();

    std::vector<double> forward(std::vector<double> x);
    void backward(std::vector<double> x, std::vector<double> y); 
    void mle_fit(
        std::vector<std::vector<double>> X, 
        std::vector<std::vector<double>> Y,
        int num_epochs,
        double lr
    );
    void reset_gradients();
    void update_params(double lr);

    double get_kinetic_energy(
        std::vector<std::vector<double>> momentum
    );

    double get_potential_energy(
        std::vector<std::vector<double>> X, 
        std::vector<std::vector<double>> Y
    );

    std::vector<std::vector<double>> hmc_step(
        std::vector<std::vector<double>> X,
        std::vector<std::vector<double>> Y,
        int num_leapfrog_steps,
        double step_size
    );

    double l2_loss(std::vector<double> y_pred, std::vector<double> y_true);  
};

#endif