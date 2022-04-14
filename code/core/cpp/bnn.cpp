#include "bnn.hpp"
#include <iostream>
using namespace std;

BNN::BNN(vector<int> layers, string activation) {
    int n = layers.size();
    int n_rows, n_cols;
    for (int l = 0; l < n - 2; l++) {
        n_rows = layers[l + 1];
        n_cols = layers[l];
        layers_.push_back(Layer(n_rows, n_cols, activation));
    }
    n_rows = layers[n - 1];
    n_cols = layers[n - 2];
    layers_.push_back(Layer(n_rows, n_cols, "linear"));
}


vector<vector<double>> BNN::get_weights() {
    vector<vector<double> > weights;
    for (Layer layer: layers_) {
        weights.push_back(layer.w_);
        weights.push_back(layer.b_);
    }
    return weights;
}

vector<vector<double>> BNN::get_gradients() {
    vector<vector<double> > gradients;
    for (Layer layer: layers_) {
        gradients.push_back(layer.w_grad_);
        gradients.push_back(layer.b_grad_);
    }
    return gradients;
}

void BNN::set_weights(vector<vector<double>> weights) {
    for (size_t l = 0; l < layers_.size(); l++) {
        layers_[l].w_ = weights[2 * l];
        layers_[l].b_ = weights[2 * l + 1];
    }
}

void BNN::mle_fit(vector<vector<double>> X, vector<vector<double>> Y, int num_epochs, double lr) {
    reset_gradients(); //Safety call to ensure gradients are all zero before training.

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (std::size_t i = 0; i < X.size(); i++) {
            vector<double> tmp = forward(X[i]);
            backward(X[i], Y[i]);
        }
        update_params(lr);
        reset_gradients();
    }
}

void BNN::update_params(double lr) {
    for (std::size_t l = 0; l < layers_.size(); l++) {
        int n_rows = layers_[l].n_rows_;
        int n_cols = layers_[l].n_cols_;
        for (int i = 0; i < n_rows; i++) {
            layers_[l].b_[i] -= lr * layers_[l].b_grad_[i];
            for (int j = 0; j < n_cols; j++) {
                layers_[l].w_[i * n_cols + j] -= lr * layers_[l].w_grad_[i * n_cols + j];
            }
        }
    } 
}

std::vector<double> BNN::forward(std::vector<double> x) {

    //Input layer:
    int n_rows = layers_[0].n_rows_;
    int n_cols = layers_[0].n_cols_;
    for (int i = 0; i < n_rows; i++) {
        double tmp = 0;
        for (int j = 0; j < n_cols; j++) {
            tmp += layers_[0].w_[i * n_cols + j] * x[j];
        }
        tmp += layers_[0].b_[i];
        layers_[0].z_[i] = tmp;
        layers_[0].a_[i] = layers_[0].act(tmp);
    }
    //The remaining layers:
    for (std::size_t l = 1; l < layers_.size(); l++) {
        n_rows = layers_[l].n_rows_;
        n_cols = layers_[l].n_cols_;
        for (int i = 0; i < n_rows; i++) {
            double tmp = 0;
            for (int j = 0; j < n_cols; j++) {
                tmp += layers_[l].w_[i * n_cols + j] * layers_[l-1].a_[j];
            }
            tmp += layers_[l].b_[i];
            layers_[l].z_[i] = tmp;
            layers_[l].a_[i] = layers_[l].act(tmp);
        }
    }
    int L = layers_.size();
    return layers_[L - 1].a_;
}

void BNN::backward(std::vector<double> x, std::vector<double> y) {
    size_t L = layers_.size(); //Number of layers.
    int n_rows, n_cols;
    double tmp;
    //Output layer.
    n_rows = layers_[L - 1].n_rows_;
    n_cols = layers_[L - 1].n_cols_;
    for (int i = 0; i < n_rows; i++) {
        tmp = layers_[L - 1].a_[i] - y[i];
        layers_[L - 1].err_[i] = tmp;

        //Compute gradients
        layers_[L - 1].b_grad_[i] += tmp;
        for (int j = 0; j < n_cols; j++) {
            layers_[L - 1].w_grad_[i * n_cols + j] += tmp * layers_[L - 2].a_[j];
        }
    }

    //Hidden layers
    for (size_t l = L - 2; l > 0; l--) {
        n_rows = layers_[l].n_rows_; //Refers to n_rows in layer l and n_cols in layer l - 1 simultaneously.
        n_cols = layers_[l].n_cols_;
        for (int j = 0; j < n_rows; j++) {
            tmp = 0.0;
            for (int k = 0; k < layers_[l+1].n_rows_; k++) {
                tmp += layers_[l+1].err_[k] * layers_[l+1].w_[k * n_rows + j];
            }
            tmp *= layers_[l].act_derivative(layers_[l].z_[j]);
            layers_[l].err_[j] = tmp;

            //Compute gradients 
            layers_[l].b_grad_[j] += tmp;
            for (int k = 0; k < n_cols; k++) {
                layers_[l].w_grad_[j * n_cols + k] += tmp * layers_[l - 1].a_[k];
            }
        }
    }

    //Input layer.
    int l = 0;
    n_rows = layers_[l].n_rows_;
    n_cols = layers_[l].n_cols_;
    for (int j = 0; j < n_rows; j++) {
        tmp = 0.0;
        for (int k = 0; k < layers_[l+1].n_rows_; k++) {
            tmp += layers_[l+1].err_[k] * layers_[l+1].w_[k * n_rows + j];
        }
        tmp *= layers_[l].act_derivative(layers_[l].z_[j]);

        //Compute gradients
        layers_[l].b_grad_[j] += tmp;
        for (int k = 0; k < n_cols; k++) {
            layers_[l].w_grad_[j * n_cols + k] += tmp * x[k];
        }
    }
}

void BNN::reset_gradients() {
    for (Layer &layer: layers_) {
        int n_rows = layer.n_rows_;
        int n_cols = layer.n_cols_;
        for (int i = 0; i < n_rows; i++) {
            layer.b_grad_[i] = 0.;
            for (int j = 0; j < n_cols; j++) {
                layer.w_grad_[i * n_cols + j] = 0.;
            }
        }
    }
}

double get_kinetic_energy(vector<vector<double>> momentum) {
    double res = 0;
    for (auto p: momentum) {
        for (size_t i = 0; i < p.size(); i++) {
            res += p[i] * p[i];
        }
    }
    return 0.5 * res;
}

double get_potential_energy(vector<vector<double>> X, vector<vector<double>> Y) {
    double res = 0;

    //L2-error (likelihood function)
    for (size_t i = 0; i < X.size(); i++) {
        vector<double> y_pred = forward(X[i]);
        res += l2_loss(y_pred, Y[i]);
        backward(X[i], Y[i]); //Add contribution to gradients.
    }

    //Regularization term (i.e prior function)
    for (Layer layer: layers_) {
        int n_rows = layer.n_rows_;
        int n_cols = layer.n_cols_;
        for (int i = 0; i < n_rows; i++) {
            res += layer.b_[i] * layer.b_[i];
            for (int j = 0; j < n_cols; j++) {
                res += layer.w_[i * n_cols + j] * layer.w_[i * n_cols + j];
            }
        }
    }
    return 0.5 * res;
}

double l2_loss(std::vector<double> y_pred, std::vector<double> y_true) {
    double res = 0;
    for (size_t i = 0; i < y_pred.size(); i++) {
        double tmp = y_pred[i] - y_true[i];
        res += tmp * tmp;
    }
    return res;
}

vector<vector<double>> hmc_step(vector<vector<double>> X, vector<vector<double>> Y, int num_leapfrog_steps, double step_size) {
    
    //Sample momentum.
    vector<vector<double>> momentum;
    for (Layer layer: layers_) {
        int n_rows = layer.n_rows_;
        int n_cols = layer.n_cols_;
        vector<double> p_w, p_b;
        for (int i = 0; i < n_rows; i++) {
            p_b.push_back(dist_(gen_));
            for (int j = 0; j < n_cols; j++) {
                p_w.push_back(dist_(gen_));
            }
        }
        momentum.push_back(p_w);
        momentum.push_back(p_b);
    }
    double K_init = get_kinetic_energy(momentum);
    double V_init = get_potential_energy(X, Y);

    double v = (uniform_real_dist_(gen_) > 0) ? 1 : -1; //Randomly choose direction in phase space.

    // Perform num_leapfrog_steps leapfrog integration in phase space.
    for (int step = 0; step < num_leapfrog_steps; step++) {

    }





    
}
