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
        l2_kernel += arma::dot(layer.w_, layer.w_);
        l2_bias += arma::dot(layer.b_, layer.b_);
    }
    return loss + l2_strength_ * (l2_kernel + l2_bias);
}





std::vector<arma::mat> MCMC::hmc_step(
    BNN model, arma::mat x, arma::mat y, int num_leapfrog_steps, double step_size
) {

    //Randomly choose direction in phase space.
    double lamb = (arma::randu() > 0.5) ? 1.0 : -1.0;

    //Get initial state
    std::vector<arma::mat> init_weights;

    std::vector<arma::mat> momentum_kernel;
    std::vector<arma::mat> momentum_bias;

    for (Layer &layer : model.layers_) {
        init_weights.push_back(layer.w_);
        init_weights.push_back(layer.b_);

        momentum_kernel.push_back(arma::mat(
            layer.w_.n_rows, layer.w_.n_cols
        ).randn()
        );

        momentum_bias.push_back(arma::mat(
            layer.b_.n_rows, layer.b_.n_cols
        ).randn()
        );
    }

    //Get initial momenta and store a copy of initial weights.
    arma::mat yhat = model.forward(x);
    double K_init = get_kinetic_energy(model, momentum_kernel, momentum_bias);
    double V_init = get_potential_energy(model.layers_, yhat, y);

    // std::cout << "Computed first K and V" << std::endl;

    model.backward(x, y);
    //First step of Leapfrog. Update momenta
    for (int l = 0; l < model.num_layers_; l++) {
        momentum_kernel[l] -= 0.5 * lamb * step_size * model.layers_[l].dw_;
        momentum_bias[l] -= 0.5 * lamb * step_size * model.layers_[l].db_;
    }


    //Inner Leapfrog steps
    for (int i = 0; i < num_leapfrog_steps - 1; i++) {

        //Update network parameters
        for (int l = 0; l < model.num_layers_; l++) {
            model.layers_[l].w_ += lamb * step_size * momentum_kernel[l];
            model.layers_[l].b_ += lamb * step_size * momentum_bias[l];
        }

        // Compute new forward and backward pass to obtain new gradients.
        arma::mat yhat = model.forward(x);
        model.backward(x, y);

        //Update momenta
        for (int l = 0; l < model.num_layers_; l++) {
            momentum_kernel[l] -= lamb * step_size * model.layers_[l].dw_;
            momentum_bias[l] -= lamb * step_size * model.layers_[l].db_;
        }
    }

    //Final step of Leapfrog

    //Update network parameters
    for (int l = 0; l < model.num_layers_; l++) {
        model.layers_[l].w_ += lamb * step_size * momentum_kernel[l];
        model.layers_[l].b_ += lamb * step_size * momentum_bias[l];
    }

    //Update momenta
    for (int l = 0; l < model.num_layers_; l++) {
        momentum_kernel[l] -= 0.5 * lamb * step_size * model.layers_[l].dw_;
        momentum_bias[l] -= 0.5 * lamb * step_size * model.layers_[l].db_;
    }

    yhat = model.forward(x);
    double K_final = get_kinetic_energy(model, momentum_kernel, momentum_bias);
    double V_final = get_potential_energy(model.layers_, yhat, y);

    double dK = K_final - K_init;
    double dV = V_final - V_init;
    double dE = dK + dV;
    // std::cout << "dE = " << dE << "\n";
    //Metropolis-Hastings
    std::vector<arma::mat> weights;
    if (dE < 0) {
        for (Layer layer : model.layers_) {
            weights.push_back(layer.w_);
            weights.push_back(layer.b_);
        }
    }
    else if (arma::randu() <= exp(-dE)){
        for (Layer layer : model.layers_) {
            weights.push_back(layer.w_);
            weights.push_back(layer.b_);
        }
    }
    else{
        for (int l = 0; l < model.num_layers_; l++) {
            model.layers_[l].w_ = init_weights[2 * l];
            model.layers_[l].b_ = init_weights[2 * l + 1];
        }
        weights = init_weights;
    }
    return weights;
}
