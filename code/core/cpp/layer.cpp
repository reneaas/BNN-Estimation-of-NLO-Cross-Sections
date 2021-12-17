#include "layer.hpp"

Layer::Layer(int n_rows, int n_cols){
    w_ = arma::mat(n_rows, n_cols).randn() * (1. / sqrt(n_cols));
    b_ = arma::mat(n_rows, 1).randn();
}
