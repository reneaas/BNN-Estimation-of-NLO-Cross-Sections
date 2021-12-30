#include "bnn.hpp"

using namespace std;

BNN::BNN(
    std::vector<Layer> layers
) {
    layers_ = layers;
    l2_strength_ = 1e-3;
}


/*
Adds a layer to the neural network.

Args:
    n_rows (int)                :   Number of units of the new layer
    n_cols (int)                :   Number of units the former layer
                                    or in case of the first layer,
                                    the former layer is the input size.

    activation (std::string)    :   Activation function. Options:
                                    `sigmoid`, `relu`.
*/
void BNN::add(int n_rows, int n_cols, std::string activation){
    layers_.push_back(Layer(n_rows, n_cols));

    if (activation == "sigmoid"){
        acts_.push_back(&BNN::sigmoid);
        acts_derivative_.push_back(&BNN::sigmoid_derivative);
    }

    else if (activation == "relu"){
        acts_.push_back(&BNN::relu);
        acts_derivative_.push_back(&BNN::relu_derivative);
    }

    else if (activation == "identity"){
        acts_.push_back(&BNN::identity);
        acts_derivative_.push_back(&BNN::identity_derivative);
    }

    num_layers_ = (int) layers_.size();
}


/*
Computes a forward pass given an input x.

Args:
    x (arma::mat)     :     Input of shape
                            [input_size, num_points]

Returns:
    x (arma::mat)     :    Result of forward pass.

*/
arma::mat BNN::forward(arma::mat x) {
    for (int l = 0; l < num_layers_; l++){
        x = layers_[l].w_ * x;
        x.each_col() += layers_[l].b_;
        layers_[l].z_ = x;

        x = (this->*acts_[l])(x);
        layers_[l].a_ = x;
    }
    return x;
}

/* 
Computes a backward pass of the network given 
given an input x and a target y.

Args:
    x (arma::mat)   :   Input features of shape [input_size, num_points]
    y (arma::mat)   :   Targets of shape [output_size, num_points]
*/
void BNN::backward(arma::mat x, arma::mat y) {
    int num_points = x.n_cols;

    //Output layer
    int l = num_layers_ - 1;
    layers_[l].err_ = layers_[l].a_ - y;
    layers_[l].dw_ = arma::mean(layers_[l - 1].a_ * layers_[l].err_.t(), 1).t() / num_points
                    + l2_strength_ * layers_[l].w_;
    layers_[l].db_ = arma::mean(layers_[l].err_, 1) + l2_strength_ * layers_[l].b_;


    for (int l = num_layers_ - 2; l >= 1; l--) {
        layers_[l].err_ = (layers_[l+1].w_.t() * layers_[l+1].err_) % layers_[l].a_ % (1 - layers_[l].a_);
        layers_[l].db_ = arma::mean(layers_[l].err_, 1) + l2_strength_ * layers_[l].b_;
        arma::mat tmp = layers_[l - 1].a_ * layers_[l].err_.t();
        tmp = arma::mean(tmp, 1);
        layers_[l].dw_ = layers_[l].err_ * layers_[l - 1].a_.t() / num_points
                        + l2_strength_ * layers_[l].w_;
    }

    //Input layer 
    l = 0; 
    layers_[l].err_ = layers_[l+1].w_.t() * layers_[l+1].err_;
    layers_[l].db_ = arma::mean(layers_[l].err_, 1) + l2_strength_ * layers_[l].b_;
    layers_[l].dw_ = arma::mean(x * layers_[l].err_.t(), 0).t() / num_points
                    + l2_strength_ * layers_[l].w_;

}

void BNN::apply_gradients() {
    for (int l = 0; l < num_layers_; l++) {
        layers_[l].w_ -= learning_rate_ * layers_[l].dw_;
        layers_[l].b_ -= learning_rate_ * layers_[l].db_;
    }
}

void BNN::mle_fit(arma::mat x, arma::mat y, int num_epochs, double learning_rate) {
    learning_rate_ = learning_rate;

    for (int i = 0; i < num_epochs; i++) {
        cout << "i = " << i << " of " << num_epochs << "\n";
        arma::mat yhat = forward(x);
        backward(x, y);
        apply_gradients();
    }
}

double BNN::loss(
    arma::mat yhat, arma::mat y
) {
    arma::mat diff = yhat - y;
    double loss = arma::dot(diff, diff);
    double l2_kernel = 0.;
    double l2_bias = 0.;
    for (auto layer : layers_) {
        l2_kernel += arma::dot(layer.w_, layer.w_);
        l2_bias += arma::dot(layer.b_, layer.b_);
    }
    return loss + l2_strength_ * (l2_kernel + l2_bias);
}

arma::mat BNN::compute_r2(arma::mat x, arma::mat y)
{   
    arma::mat yhat = forward(x);

    arma::mat diff = yhat - y;
    arma::mat y_mean = arma::mean(y, 1);
    diff = diff % diff;
    arma::mat err = arma::sum(diff, 1);
    diff = y - y_mean(0);
    arma::mat tmp = arma::sum(diff % diff, 1);
    arma::mat r2 = 1 - err / tmp;
    return r2;
}


/*  
Hidden layer activation functions
*/
arma::mat BNN::sigmoid(arma::mat z) {
    return 1./(1. + exp(-z));
}

arma::mat BNN::sigmoid_derivative(arma::mat z) {
    arma::vec res = 1./(1. + exp(-z));
    return res % (1-res);
}


arma::mat BNN::relu(arma::mat z) {
    arma::mat s = z;
    return s.transform( [](double val){return val*(val > 0);});
}

arma::mat BNN::relu_derivative(arma::mat z) {
    arma::mat s = z;
    return s.transform( [](double val){return (val > 0);});
}

arma::mat BNN::identity(arma::mat z) {
    return z;
}

arma::mat BNN::identity_derivative(arma::mat z) {
    return arma::vec(size(z)).fill(1.);
}