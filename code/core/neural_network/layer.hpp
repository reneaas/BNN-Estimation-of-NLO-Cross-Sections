#ifndef LAYER_HPP
#define LAYER_HPP

#include <armadillo>


/*
 The Layer class is used by the FFNN class to store properties of each layer.
*/
class Layer {
public:

    int rows_, cols_; //Number of rows and columns in the weights_ matrix.

    //Properties a layer in the neural net.
    arma::vec bias_;
    arma::vec error_;
    arma::vec activation_;
    arma::mat weights_;
    arma::vec z_;

    //Store gradients
    arma::mat dw_;
    arma::vec db_;

    //Used with sgd_momentum
    arma::mat w_mom_;
    arma::vec b_mom_;


    //Constructors
    Layer(int rows, int cols);
    Layer(int rows, int cols, std::string optimizer);
};

#endif
