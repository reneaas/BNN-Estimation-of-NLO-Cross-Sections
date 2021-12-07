#include "layer.hpp"

Layer::Layer(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    weights_ = arma::randn(rows_, cols_)*(1./sqrt(cols_));
    bias_ = arma::randn(rows_);
    activation_ = arma::vec(rows_).fill(0.);
    z_ = arma::vec(rows_).fill(0.);
    error_ = arma::vec(rows_).fill(0.);

    //Initialize gradients
    dw_ = arma::mat(rows_, cols_).fill(0.);
    db_ = arma::vec(rows_).fill(0.);
}

Layer::Layer(int rows, int cols, std::string optimizer) {
    rows_ = rows;
    cols_ = cols;
    weights_ = arma::randn(rows_, cols_)*(1./sqrt(cols_));
    bias_ = arma::randn(rows_);
    activation_ = arma::vec(rows_).fill(0.);
    z_ = arma::vec(rows_).fill(0.);
    error_ = arma::vec(rows_).fill(0.);

    //Initialize gradients
    dw_ = arma::mat(rows_, cols_).fill(0.);
    db_ = arma::vec(rows_).fill(0.);

    if (optimizer == "sgd_momentum"){
        w_mom_ = arma::mat(rows_, cols_).fill(0.);
        b_mom_ = arma::vec(rows_).fill(0.);
    }

}
