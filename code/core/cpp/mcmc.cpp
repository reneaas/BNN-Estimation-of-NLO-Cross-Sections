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
        sum += arma::dot(p_w[i], p_w[i]);
        sum += arma::dot(p_b[i], p_b[i]);
    }
    return sum;
}

// double MCMC::get_potential_energy(
//     std::vector<Layer> layers, arma::mat yhat, arma::mat y
// ) {
//     arma::mat diff = yhat - y;
//     double loss = arma::dot(diff, diff);
//     double l2_kernel = 0.;
//     double l2_bias = 0.;
//     for (auto layer : layers) {
//         l2_kernel += arma::dot(layer.w_, layer.w_);
//         l2_bias += arma::dot(layer.b_, layer.b_);
//     }
//     return loss + l2_strength_ * (l2_kernel + l2_bias);
// }



// std::vector<std::vector<arma::mat>> MCMC::sample_chain_hmc(
//     std::vector<Layer> layers, arma::mat x, arma::mat y, int num_results, int num_leapfrog_steps, double step_size
// ) {
//     BNN model = BNN(layers);
//     std::cout << "Got here" << std::endl;
//     std::vector<std::vector<arma::mat>> samples;
//     // samples.reserve(num_results);
//     std::vector<arma::mat> tmp;
//     std::cout << "Got here" << std::endl;
//     for (int i = 0; i < num_results; i++) {
//         std::cout << "Got here" << std::endl;
//         std::cout << "i = " << i << " / " << num_results << std::endl;
//         std::cout << "Got here" << std::endl;
//         tmp = hmc_step(model, x, y, num_leapfrog_steps, step_size);
//         // samples.push_back();
//         std::cout << "Got here" << std::endl;
//     }
//     std::cout << " " << std::endl;
//     return samples;
// }


std::vector<arma::mat> MCMC::hmc_step(
    BNN *model, arma::mat x, arma::mat y, int num_leapfrog_steps, double step_size
) {
    //Randomly choose direction in phase space.
    double lamb = (arma::randu() > 0.5) ? 1.0 : 0.0;

    //Get initial state
    std::vector<arma::mat> init_kernel;
    std::vector<arma::mat> init_bias;
    // std::cout << "init_kernel size= " << init_kernel.size() << std::endl;

    //Gibbs sampler
    std::vector<arma::mat> momentum_kernel;
    std::vector<arma::mat> momentum_bias;
    // momentum_kernel.reserve(model.num_layers_);
    // momentum_bias.reserve(model.num_layers_);


    //Get initial momenta and store a copy of initial weights.
    for (auto layer : model->layers_) {
        //Store weights
        init_kernel.push_back(layer.w_);
        init_bias.push_back(layer.b_);

        //Create generalized momenta.
        momentum_kernel.push_back(arma::mat(
            layer.w_.n_rows, layer.w_.n_cols
        ).randn()
        );
        momentum_bias.push_back(arma::mat(
            layer.b_.n_rows, layer.b_.n_cols
        ).randn()
        );
    }

    // std::cout << "Create momenta and stored old weights" << std::endl;


    arma::mat yhat = model->forward(x);
    double K_init = get_kinetic_energy(*model, momentum_kernel, momentum_bias);
    //double V_init = get_potential_energy(model.layers_, yhat, y);
    double V_init = model->loss(yhat, y);

    model->backward(x, y);
    //First step of Leapfrog. Update momenta
    for (int l = 0; l < model->num_layers_; l++) {
        momentum_kernel[l] -= 0.5 * lamb * step_size * model->layers_[l].dw_;
        momentum_bias[l] -= 0.5 * lamb * step_size * model->layers_[l].db_;
    }

    //Inner Leapfrog steps
    for (int i = 0; i < num_leapfrog_steps - 1; i++) {

        //Update network parameters
        for (int l = 0; l < model->num_layers_; l++) {
            model->layers_[l].w_ += lamb * step_size * momentum_kernel[l];
            model->layers_[l].b_ += lamb * step_size * momentum_bias[l];
        }

        // Compute new forward and backward pass to obtain new gradients.
        arma::mat yhat = model->forward(x);
        model->backward(x, y);

        //Update momenta
        for (int l = 0; l < model->num_layers_; l++) {
            momentum_kernel[l] -= lamb * step_size * model->layers_[l].dw_;
            momentum_bias[l] -= lamb * step_size * model->layers_[l].db_;
        }
    }

    //Final step of Leapfrog

    //Update network parameters
    for (int l = 0; l < model->num_layers_; l++) {
        model->layers_[l].w_ += lamb * step_size * momentum_kernel[l];
        model->layers_[l].b_ += lamb * step_size * momentum_bias[l];
    }

    //Update momenta
    for (int l = 0; l < model->num_layers_; l++) {
        momentum_kernel[l] -= 0.5 * lamb * step_size * model->layers_[l].dw_;
        momentum_bias[l] -= 0.5 * lamb * step_size * model->layers_[l].db_;
    }

    yhat = model->forward(x);
    double K_final = get_kinetic_energy(*model, momentum_kernel, momentum_bias);
    double V_final = model->loss(yhat, y);

    double dK = K_final - K_init;
    double dV = V_final - V_init;
    double dE = dK + dV;

    //Metropolis-Hastings
    std::vector<arma::mat> weights;
    if (dE < 0) {
        for (auto layer : model->layers_) {
            weights.push_back(layer.w_);
            weights.push_back(layer.b_);
        }
    }
    else if (arma::randu() <= exp(-dE)) {
        for (auto layer : model->layers_) {
            weights.push_back(layer.w_);
            weights.push_back(layer.b_);
        }
    }
    else{
        //Reject sample and set network weights to init weights.
        // weights = init_weights;
        for (int l = 0; l < model->num_layers_; l++) {
            weights.push_back(init_kernel[l]);
            weights.push_back(init_bias[l]);
            model->layers_[l].w_ = init_kernel[l];
            model->layers_[l].b_ = init_bias[l];
    }
    }

    return weights;
}