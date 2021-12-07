#include "neural_network.hpp"

/*
Constructor that sets up a basic model with no layers. Meant to be used with the member function add_layer(int rows, int cols).
*/
FFNN::FFNN(int features, int num_outputs, std::string model_type, double lamb, double gamma, std::string hidden_activation) {
    features_ = features;
    num_outputs_ = num_outputs;

    //Set top layer activation function and which metric to use.
    if (model_type == "classification"){
        if (num_outputs == 1){
            top_layer_act = &FFNN::binary_classifier;
        }
        else{
            top_layer_act = &FFNN::softmax; //top layer activation
        }
        compute_metric = &FFNN::compute_accuracy;
    }
    else if (model_type == "regression"){
        top_layer_act = &FFNN::linear;
        compute_metric = &FFNN::compute_r2;
    }

    //Set hidden activation function
    if (hidden_activation == "sigmoid"){
        hidden_act = &FFNN::sigmoid;
        hidden_act_derivative = &FFNN::sigmoid_derivative;
    }
    else if (hidden_activation == "relu"){
        hidden_act = &FFNN::relu;
        hidden_act_derivative = &FFNN::relu_derivative;
    }
    else if (hidden_activation == "leaky_relu"){
        hidden_act = &FFNN::leaky_relu;
        hidden_act_derivative = &FFNN::leaky_relu_derivative;
    }

    gamma_ = gamma;
    lamb_ = lamb;

    if (gamma_ > 0){
        update_parameters = &FFNN::update_l2_momentum;
    }
    else if (gamma_ == 0 && lamb_ > 0){
        update_parameters = &FFNN::update_l2;
    }
    else{
        update_parameters = &FFNN::update;
    }
}

/*
Constructor that sets up a neural net with equally many hidden neurons in each hidden layer.
*/
FFNN::FFNN(int hidden_layers, int features, int nodes, int num_outputs, std::string model_type, double lamb, double gamma, std::string hidden_activation) {
    features_ = features;
    nodes_ = nodes;
    num_outputs_ = num_outputs;
    hidden_layers_ = hidden_layers;
    num_layers_ = hidden_layers_ + 1;

    //Set top layer activation function and which metric to use.
    if (model_type == "classification"){
        if (num_outputs == 1){
            top_layer_act = &FFNN::binary_classifier;
        }
        else{
            top_layer_act = &FFNN::softmax; //top layer activation
        }
        compute_metric = &FFNN::compute_accuracy;
    }
    else if (model_type == "regression"){
        top_layer_act = &FFNN::linear;
        compute_metric = &FFNN::compute_r2;
    }

    //Set hidden activation function
    if (hidden_activation == "sigmoid"){
        hidden_act = &FFNN::sigmoid;
        hidden_act_derivative = &FFNN::sigmoid_derivative;
    }
    else if (hidden_activation == "relu"){
        hidden_act = &FFNN::relu;
        hidden_act_derivative = &FFNN::relu_derivative;
    }
    else if (hidden_activation == "leaky_relu"){
        hidden_act = &FFNN::leaky_relu;
        hidden_act_derivative = &FFNN::leaky_relu_derivative;
    }

    gamma_ = gamma;
    lamb_ = lamb;

    if (gamma_ > 0){
        update_parameters = &FFNN::update_l2_momentum;

        //Add first hidden layer
        layers_.push_back(Layer(nodes_, features_, "sgd_momentum"));

        //Add hidden layers
        for (int l = 1; l < hidden_layers_; l++){
            layers_.push_back(Layer(nodes_, nodes_, "sgd_momentum"));
        }

        //Add top layer
        layers_.push_back(Layer(num_outputs_, nodes_, "sgd_momentum"));
    }
    else if (gamma_ == 0 && lamb_ > 0){
        update_parameters = &FFNN::update_l2;

        //Add first hidden layer
        layers_.push_back(Layer(nodes_, features_));

        //Add hidden layers
        for (int l = 1; l < hidden_layers_; l++){
            layers_.push_back(Layer(nodes_, nodes_));
        }

        //Add top layer
        layers_.push_back(Layer(num_outputs_, nodes_));
    }
    else{
        update_parameters = &FFNN::update;

        update_parameters = &FFNN::update_l2;

        //Add first hidden layer
        layers_.push_back(Layer(nodes_, features_));

        //Add hidden layers
        for (int l = 1; l < hidden_layers_; l++){
            layers_.push_back(Layer(nodes_, nodes_));
        }

        //Add top layer
        layers_.push_back(Layer(num_outputs_, nodes_));
    }
}


/*
Adds a layer to the neural net.
Parameters:
int rows: number of neurons in the layer
int cols: number of neurons in the previous layer.
*/
void FFNN::add_layer(int rows, int cols) {
    if (gamma_ > 0){
        layers_.push_back(Layer(rows, cols, "sgd_momentum"));
    }
    else{
        layers_.push_back(Layer(rows, cols));
    }
    num_layers_ = layers_.size();
    hidden_layers_ = num_layers_ - 1;
}



void FFNN::init_data(arma::mat X_train, arma::mat y_train) {
    num_points_ = X_train.n_cols;
    X_train_ = X_train;
    y_train_ = y_train;
}

void FFNN::fit(int epochs, int batch_sz, double eta) {
    batch_sz_ = batch_sz;
    eta_ = eta;
    //epochs_ = epochs;
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, num_points_-1);
    int idx;
    int batches = num_points_/batch_sz_;
    arma::vec x, y;
    x = arma::vec(features_);
    y = arma::vec(num_outputs_);
    for (int epoch = 0; epoch < epochs; epoch++){
        std::cout << " epoch = " << epoch+1 << " of " << epochs << std::endl;

        for (int batch = 0; batch < batches; batch++){

            for (int b = 0; b < batch_sz_; b++){
                idx = distribution(generator);
                x = X_train_.col(idx);
                y = y_train_.col(idx);
                feed_forward(x);
                backward_pass(x, y);
            }
            (this->*update_parameters)();
        }
    }
}

/*
Basic update rule for the gradients with no regularization or momentum. Computed on a batch.
*/
void FFNN::update() {
    double step = eta_*(1./batch_sz_);

    for (int l = 0; l < num_layers_; l++){
        layers_[l].weights_ -=  step*layers_[l].dw_;
        layers_[l].bias_ -= step*layers_[l].db_;
        layers_[l].dw_.fill(0.);
        layers_[l].db_.fill(0.);
    }
}

/*
Update rule for the gradients with regularization. No momentum. Computed on a batch.
*/
void FFNN::update_l2() {
    double step = eta_*(1./batch_sz_);
    for (int l = 0; l < num_layers_; l++){
        layers_[l].weights_ -= step*(lamb_*layers_[l].weights_ + layers_[l].dw_);
        layers_[l].bias_ -= step*layers_[l].db_;
        layers_[l].dw_.fill(0.);
        layers_[l].db_.fill(0.);
    }
}

/*
Update rule for the gradients with regularization and momentum over a batch.
*/
void FFNN::update_l2_momentum() {
    double step = eta_*(1./batch_sz_);
    for (int l = 0; l < num_layers_; l++){
        layers_[l].w_mom_ = gamma_*layers_[l].w_mom_ +  step*(lamb_*layers_[l].weights_ + layers_[l].dw_);
        layers_[l].weights_ -= layers_[l].w_mom_;
        layers_[l].b_mom_ = gamma_*layers_[l].b_mom_ + step*layers_[l].db_;
        layers_[l].bias_ -= layers_[l].b_mom_;
        layers_[l].dw_.fill(0.);
        layers_[l].db_.fill(0.);
    }
}

/*
Feed-forward part of the backpropagation algorithm
*/
void FFNN::feed_forward(arma::vec x) {
    //Process input activation
    layers_[0].z_ = layers_[0].weights_*x + layers_[0].bias_;
    layers_[0].activation_ = (this->*hidden_act)(layers_[0].z_);

    //Hidden layer activations
    for (int l = 1; l < hidden_layers_; l++){
        layers_[l].z_ = layers_[l].weights_*layers_[l-1].activation_ + layers_[l].bias_;
        layers_[l].activation_ = (this->*hidden_act)(layers_[l].z_);
    }

    //Top layer activation
    int l = num_layers_-1;
    layers_[l].z_ = layers_[l].weights_*layers_[l-1].activation_ + layers_[l].bias_;
    layers_[l].activation_ = (this->*top_layer_act)(layers_[l].z_);
}

/*
Backward pass of the backpropagation algorithm
*/
void FFNN::backward_pass(arma::vec x, arma::vec y) {
    //Top layer error
    int l = num_layers_-1;
    layers_[l].error_ = layers_[l].activation_ - y;
    add_gradients(l);


    //Error in hidden layers
    for (int l = num_layers_ - 2; l > 0 ; l--){
        layers_[l].error_ = (layers_[l+1].weights_.t() * layers_[l+1].error_) % (this->*hidden_act_derivative)(layers_[l].z_);
        add_gradients(l);
    }

    //Compute gradients in first layer.
    l = 0;
    layers_[l].error_ = (layers_[l+1].weights_.t() * layers_[l+1].error_) % (this->*hidden_act_derivative)(layers_[l].z_);
    layers_[l].db_ += layers_[l].error_;
    layers_[l].dw_ += layers_[l].error_*x.t();

}

/*
Evaluate the model on validation or test data
*/
double FFNN::evaluate(arma::mat X_test, arma::mat y_test) {
    X_test_ = X_test;
    y_test_ = y_test;
    num_test_ = X_test_.n_cols;

    double metric = (this->*compute_metric)();
    return metric;
}

/*
Computes accuracy on the test set. Pointer to by compute_metric if model_type = "classification".
*/
double FFNN::compute_accuracy() {
    arma::vec x = arma::vec(features_);
    arma::vec y = arma::vec(num_outputs_);

    int idx;
    double accuracy = 0;
    double correct_predictions = 0;
    double wrong_predictions = 0;
    int l = num_layers_-1;
    for (int i = 0; i < num_test_; i++){
        x = X_test_.col(i);
        y = y_test_.col(i);
        feed_forward(x);
        idx = index_max(layers_[l].activation_);
        if (y(idx) == 1){
            correct_predictions++;
        }
    }
    accuracy = correct_predictions*(1./num_test_);
    return accuracy;
}

/*
Computes R2 score on a test set. Pointed to by compute_metric if model_type = "regression"
*/
double FFNN::compute_r2() {
    arma::vec x = arma::vec(features_);
    arma::vec y = arma::vec(num_outputs_);
    double diff;

    double error = 0.;
    double y_mean = 0.;
    int l = num_layers_-1;
    for (int i = 0; i < num_test_; i++){
        x = X_test_.col(i);
        y = y_test_.col(i);
        feed_forward(x);
        diff = y(0) - layers_[l].activation_(0);
        error += diff*diff;
        y_mean += y(0);
    }
    y_mean *= (1./num_test_);
    double tmp = 0;

    for (int j = 0; j < num_test_; j++){
        tmp += (y_test_(0, j)-y_mean)*(y_test_(0, j)-y_mean);
    }
    double r2 = 1 - error/tmp;
    return r2;
}

/*
Computes mse, but is by default no used.
Can manually be changed in the constructor.
*/
double FFNN::compute_mse() {
    arma::vec x = arma::vec(features_);
    arma::vec y = arma::vec(num_outputs_);

    double mse = 0.;
    double diff;
    int l = num_layers_-1;
    for (int i = 0; i < num_test_; i++){
        x = X_test_.col(i);
        y = y_test_.col(i);
        feed_forward(x);

        diff = y(0) - layers_[l].activation_(0);
        mse += diff*diff;
    }
    mse *= (1./num_test_);
    return mse;
}


void FFNN::add_gradients(int l) {
    layers_[l].db_ += layers_[l].error_;
    layers_[l].dw_ += layers_[l].error_*layers_[l-1].activation_.t(); //outer product
}

/*
Hidden layer activation functions
*/
arma::vec FFNN::sigmoid(arma::vec z) {
    return 1./(1. + exp(-z));
}

arma::vec FFNN::sigmoid_derivative(arma::vec z) {
    arma::vec res = 1./(1. + exp(-z));
    return res % (1-res);
}


arma::vec FFNN::relu(arma::vec z) {
    arma::vec s = z;
    return s.transform( [](double val){return val*(val > 0);});
}

arma::vec FFNN::relu_derivative(arma::vec z) {
    arma::vec s = z;
    return s.transform( [](double val){return (val > 0);});
}

arma::vec FFNN::leaky_relu(arma::vec z) {
    arma::vec s = z;
    return s.transform( [](double val){return 0.01*val*(val <= 0) + val*(val > 0);});
}

arma::vec FFNN::leaky_relu_derivative(arma::vec z) {
    arma::vec s = z;
    return s.transform( [](double val){return 0.01*(val <= 0) + (val > 0);});
}

/*
Top layer activation functions
*/
arma::vec FFNN::softmax(arma::vec a) {
    arma::vec res = exp(a);
    return exp(a)/sum(res);
}

arma::vec FFNN::linear(arma::vec z) {
    return z;
}

arma::vec FFNN::binary_classifier(arma::vec z) {
    arma::vec s = 1./(1.+exp(z));
    return s.transform( [](double val){return (val > 0.5);});
}
