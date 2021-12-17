#include "mcmc.hpp"

MCMC::MCMC() {
    l2_strength_ = 1e-3;
}

MCMC::MCMC(double l2_strength) {
    l2_strength_ = l2_strength;
}


double MCMC::get_kinetic_energy(
    BNN model, std::vector<arma::mat> p_w, std::vector<arma::mat> p_b
) {

    double sum = 0.;
    for (int i = 0; i < model.num_layers_; i++) {
        arma::mat w = p_w.at(i);
        sum += arma::dot(p_w[i], p_w[i]);
        sum += arma::dot(p_b[i], p_b[i]);
    }

    // for (arma::mat p : p_w) {
    //     std::cout << "Got here" << std::endl;
    //     sum += arma::dot(p, p);
    // }
    
    // for (arma::mat p : p_b) {
    //     sum += arma::dot(p, p);
    // }

    return sum;
}

double MCMC::get_potential_energy(
    std::vector<Layer> layers, arma::mat yhat, arma::mat y
) {
    arma::mat diff = yhat - y;
    double loss = arma::dot(diff, diff);
    double l2_kernel = 0.;
    double l2_bias = 0.;
    for (Layer layer : layers) {
        // layer.w_.print("w = ");
        l2_kernel += arma::dot(layer.w_, layer.w_);
        l2_bias += arma::dot(layer.b_, layer.b_);
    }
    return loss + l2_strength_ * (l2_kernel + l2_bias);
}





std::vector<std::vector<arma::mat>> MCMC::hmc_step(
    BNN model, arma::mat x, arma::mat y, int num_leapfrog_steps, double step_size
) {

    //Randomly choose direction in phase space.
    double lamb = (arma::randu() > 0.5) ? 1.0 : 0.0;

    //Get initial state
    std::vector<arma::mat> init_kernel;
    std::vector<arma::mat> init_bias;
    init_kernel.reserve(model.num_layers_);
    init_bias.reserve(model.num_layers_);

    //Gibbs sampler
    std::vector<arma::mat> momentum_kernel;
    std::vector<arma::mat> momentum_bias;
    // momentum_kernel.reserve(model.num_layers_);
    // momentum_bias.reserve(model.num_layers_);


    //Get initial momenta and store a copy of initial weights.
    for (int l = 0; l < model.num_layers_; l++) {
        Layer &layer = model.layers_[l];
        init_kernel[l] = layer.w_;
        init_bias[l] = layer.b_;

        momentum_kernel.push_back(arma::mat(
            layer.w_.n_rows, layer.w_.n_cols
        ).randn()
        );

        momentum_bias.push_back(arma::mat(
            layer.b_.n_rows, layer.b_.n_cols
        ).randn()
        );
    }
    arma::mat yhat = model.forward(x);
    double K_init = get_kinetic_energy(model, momentum_kernel, momentum_bias);
    double V_init = get_potential_energy(model.layers_, yhat, y);

    std::cout << "Computed first K and V" << std::endl;

    model.backward(x, y);
    //First step of Leapfrog. Update momenta
    for (int l = 0; l < model.num_layers_; l++) {
        momentum_kernel[l] -= 0.5 * step_size * model.layers_[l].dw_;
        momentum_bias[l] -= 0.5 * step_size * model.layers_[l].db_;
    }

    std::cout << "Passed first Leapfrog step" << std::endl;

    //Inner Leapfrog steps
    for (int i = 0; i < num_leapfrog_steps - 1; i++) {

        //Update network parameters
        for (int l = 0; l < model.num_layers_; l++) {
            model.layers_[l].w_ += step_size * momentum_kernel[l];
            model.layers_[l].b_ += step_size * momentum_bias[l];
        }

        // Compute new forward and backward pass to obtain new gradients.
        arma::mat yhat = model.forward(x);
        model.backward(x, y);

        //Update momenta
        for (int l = 0; l < model.num_layers_; l++) {
            momentum_kernel[l] -= step_size * model.layers_[l].dw_;
            momentum_bias[l] -= step_size * model.layers_[l].db_;
        }
    }

    //Final step of Leapfrog

    //Update network parameters
    for (int l = 0; l < model.num_layers_; l++) {
        model.layers_[l].w_ += step_size * momentum_kernel[l];
        model.layers_[l].b_ += step_size * momentum_bias[l];
    }

    //Update momenta
    for (int l = 0; l < model.num_layers_; l++) {
        momentum_kernel[l] -= 0.5 * step_size * model.layers_[l].dw_;
        momentum_bias[l] -= 0.5 * step_size * model.layers_[l].db_;
    }

    yhat = model.forward(x);
    double K_final = get_kinetic_energy(model, momentum_kernel, momentum_bias);
    double V_final = get_potential_energy(model.layers_, yhat, y);

    double dK = K_final - K_init;
    double dV = V_final - V_init;
    double dE = dK + dV;
    double p = exp(-dV) * exp(-dK);

    //Metropolis-Hastings
    std::vector<arma::mat> kernel, bias;
    if (dE < 0) {
        for (Layer layer : model.layers_) {
            kernel.push_back(layer.w_);
            bias.push_back(layer.b_);
        }
    }
    else if (arma::randu() <= exp(-dV) * exp(-dK)) {
        for (Layer layer : model.layers_) {
            kernel.push_back(layer.w_);
            bias.push_back(layer.b_);
        }
    }
    else{
        kernel = init_kernel;
        bias = init_bias;
    }
    std::vector<std::vector<arma::mat>> res = {kernel, bias};
    return res;
}