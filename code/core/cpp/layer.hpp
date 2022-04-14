#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <functional>
#include <string>
#include <random>
#include <cmath>

class Layer{

private:
    std::mt19937_64 gen_;
    std::normal_distribution<double> dist_;

public:
    Layer(){}; //Nullary constructor
    Layer(int n_rows, int n_cols, std::string activation);
    virtual ~Layer(){};


    std::vector<double> w_, b_, z_, a_;
    std::vector<double> w_grad_, b_grad_, err_;
    int n_rows_, n_cols_;
    std::function<double(double)> act;
    std::function<double(double)> act_derivative;

    //Activation functions
    double sigmoid(double x);
    double sigmoid_derivative(double x);

    double relu(double x);
    double relu_derivative(double x);

    double tanh(double x);
    double tanh_derivative(double x);

    //Utility functions
    std::vector<double> get_kernel();
    std::vector<double> get_bias();
    

};

#endif

