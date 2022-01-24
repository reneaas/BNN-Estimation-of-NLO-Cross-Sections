#ifndef LAYER_HPP
#define LAYER_HPP

#include <armadillo>

class Layer {
private:
    /* data */

public:
    Layer (int n_cols, int n_rows);
    virtual ~Layer (){};
    arma::mat w_;
    arma::mat b_;
    arma::mat z_;
    arma::mat a_;
    arma::mat err_;

    arma::mat dw_;
    arma::mat db_;
};

#endif
