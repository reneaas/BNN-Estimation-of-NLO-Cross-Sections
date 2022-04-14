#include "layer.hpp"

using namespace std;

Layer::Layer(int n_rows, int n_cols, string activation) {
    n_rows_ = n_rows;
    n_cols_ = n_cols;
    w_ = std::vector<double>(n_rows * n_cols, 0.0);
    b_ = std::vector<double>(n_rows, 0.0);
    z_ = std::vector<double>(n_rows, 0.0);
    a_ = std::vector<double>(n_rows, 0.0);

    w_grad_ = std::vector<double>(n_rows * n_cols, 0.0);
    b_grad_ = std::vector<double>(n_rows, 0.0);
    err_ = std::vector<double>(n_rows, 0.0);

    for (int i = 0; i < n_rows; i++) {
        b_[i] = dist_(gen_);
        for (int j = 0; j < n_cols; j++) {
            w_[i * n_cols + j] = dist_(gen_) / sqrt(n_cols);
        }
    }

    if (activation == "sigmoid") {
        act = [&](double x) {
            return sigmoid(x);
        };

        act_derivative = [&](double x) {
            return sigmoid_derivative(x);
        };
    }
    else if (activation == "relu") {
        act = [&](double x) {
            return relu(x);
        };

        act_derivative = [&](double x) {
            return relu_derivative(x);
        };
    }
    else if (activation == "tanh") {
        act = [&](double x) {
            return tanh(x);
        };

        act_derivative = [&](double x) {
            return tanh_derivative(x);
        };
    }

    else if (activation == "linear"){
        act = [&](double x) {
            return x;
        };

        act_derivative = [&](double x) {
            return 1.;
        };
    }
}


vector<double> Layer::get_kernel() {
    return w_;
}

vector<double> Layer::get_bias() {
    return b_;
}


double Layer::sigmoid(double x) {
    return 1. / (1 + exp(-x));
}

double Layer::sigmoid_derivative(double x) {
    double tmp = sigmoid(x);
    return tmp * (1 - tmp);
}

double Layer::relu(double x) {
    return x * (x > 0);
}

double Layer::relu_derivative(double x) {
    return 1. * (x > 0);
}

double Layer::tanh(double x) {
    return std::tanh(x);
}

double Layer::tanh_derivative(double x) {
    double tmp = std::tanh(x);
    return 1 - tmp * tmp;
}