

g++-11 -c neural_network.cpp layer.cpp regression_main.cpp -O3
g++-11 -o regression_main.out *.o -larmadillo
./regression_main.out
