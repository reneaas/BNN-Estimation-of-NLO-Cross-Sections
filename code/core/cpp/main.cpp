#include "layer.hpp"
#include "bnn.hpp"
#include "mcmc.hpp"
#include <armadillo>
#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include <omp.h>

arma::mat sigmoid(arma::mat z) {
    return 1./(1. + exp(-z));
}

arma::mat f(arma::mat x) {
    arma::mat y = arma::mat(1, x.n_cols);
    for (int i = 0; i < (int) x.n_cols; i++){
        y(0, i) = sin(x(0, i));
    }
    return y;
}

using namespace std;
using namespace arma;

BNN get_model(){
    BNN bnn = BNN();
    bnn.add(2500, 1, "relu");
    // bnn.add(100, 250, "relu");
    bnn.add(1, 2500, "identity");
    return bnn;
}

std::vector<Layer> get_layer() {
    vector<Layer> layers;
    layers.push_back(Layer(5000, 1));
    layers.push_back(Layer(250, 5000));
    layers.push_back(Layer(1, 250));
    return layers;
}

std::vector<std::string> get_activations() {
    vector<std::string> activations;
    activations.push_back("sigmoid");
    activations.push_back("identity");
    return activations;
}

/* 
TODO: Add burn-in steps to the sample chain.
*/
std::vector<std::vector<arma::mat>> sample_chain(
    arma::mat x, arma::mat y, int num_results, int num_leapfrog_steps, int num_burnin_steps, double step_size
) {
    std::vector<std::vector<arma::mat>> samples;
    #ifdef _OPENMP
    {   
        #pragma omp parallel num_threads(2)
        {
            int id = omp_get_thread_num();
            arma::arma_rng::set_seed(id + 42);
            BNN model = get_model();
            MCMC mcmc = MCMC();

            for (int i = 0; i < num_burnin_steps; i++) {
                std::vector<arma::mat> tmp = mcmc.hmc_step(
                    &model, x, y, num_leapfrog_steps, step_size
                );
            }
            int local_num_results = num_results / omp_get_num_threads();
            std::vector<std::vector<arma::mat>> samples_local;
            #pragma omp for 
            for (int i = 0; i < num_results; i++) {
                if (id == 0) {
                    cout << "\r progress = " << (1.*i / local_num_results) * 100 << " %" << std::flush;
                }

                samples_local.push_back(
                        mcmc.hmc_step(&model, x, y, num_leapfrog_steps, step_size
                    )
                );
            }

            #pragma omp critical
            {  
                for (auto sample : samples_local) {
                    samples.push_back(sample);
                }
            }
        }
    }
    #else
    {
        BNN model = get_model();
        MCMC mcmc = MCMC();

        for (int i = 0; i < num_burnin_steps; i++) {
                std::vector<arma::mat> tmp = mcmc.hmc_step(
                    &model, x, y, num_leapfrog_steps, step_size
                );
            }

        for (int i = 0; i < num_results; i++) {
            cout <<  "\r" << "i = " << i << " / " << num_results;
            samples.push_back(
                    mcmc.hmc_step(&model, x, y, num_leapfrog_steps, step_size
                )
            );
        }
        cout << endl;
    }
    #endif

    return samples;
}


int main(int argc, char const *argv[]) {

    
    int n_train = 1000;
    int dims = 1;
    arma::mat x_train = arma::mat(dims, n_train).randn();
    arma::mat y_train = f(x_train);

    // BNN model = get_model();

    // arma::mat yhat = bnn.forward(x_train);
    // //yhat.print("yhat = ");
    // bnn.backward(x_train, y_train);

    // int num_epochs = 1000;
    // double learning_rate = 0.01;

    // bnn.mle_fit(x_train, y_train, num_epochs, learning_rate);
    // arma::mat r2 = bnn.compute_r2(x_train, y_train);
    // cout << "r2 = " << r2 << endl;

    // MCMC mcmc = MCMC();

    // std::vector<arma::mat> res;
    // model.layers_[0].w_.print("w0 before = ");
    // res = mcmc.hmc_step(&model, x_train, y_train, 60, 0.001);
    int num_results = 1000;
    int num_burnin_steps = 0;
    int num_leapfrog_steps = 60;
    double step_size = 0.001;
    std::vector<std::vector<arma::mat>> samples;
    double start = omp_get_wtime();
    samples = sample_chain(
        x_train, y_train, num_results, num_leapfrog_steps, num_burnin_steps, step_size
    );
    double end = omp_get_wtime();
    double timeused = end - start;
    cout << "timeused = " << timeused << " seconds" << endl;


    // for (auto neural_net : samples) {
    //     for (std::size_t i = 0; i < neural_net.size(); i++) {

    //     }
    // }

    std::string path = "./bnn_samples/";
    for (std::size_t i = 0; i < samples.size(); i++) {
        for (std::size_t j = 0; j < samples[i].size(); j++) {
            std::string fname = path
                                + "weights" 
                                + to_string(i)
                                + "_"
                                + to_string(j)
                                + ".bin";
            
            samples[i][j].save(fname, arma::arma_binary);
            //samples[i][j].print("weight = ");
        }
    }

    

    // int n = 2500;
    // int m = 1000;

    // double *a = new double[n * m]();
    // double *b = new double[m * n]();
    // double *c = new double[n * n]();

    // for (int i = 0; i < n * m; i++){
    //     a[i] = arma::randn();
    //     b[i] = arma::randn();
    // }


    // double start = omp_get_wtime();
    // #pragma omp parallel for collapse(3)
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         for (int k = 0; k < m; k++) {
    //             c[i * n + j] += a[i * m + k] * b[k * n + j];
    //         }
    //     } 
    // }
    // double end = omp_get_wtime();
    // double timeused = end - start;
    // std::cout << "Timeused = " << timeused << " seconds" << std::endl;

    // delete[] a;
    // delete[] b;
    // delete[] c;

    // arma::mat A = arma::mat(n, m).randn();
    // arma::mat B = arma::mat(m, n).randn();
    // start = omp_get_wtime();
    // arma::mat C = A * B;
    // end = omp_get_wtime();
    // timeused = end - start;
    // std::cout << "Timeused = " << timeused << " seconds" << std::endl;


    return 0;
}

