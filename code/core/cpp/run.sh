
if [[ "$1" == "serial" ]]
then
    g++-11 -c main.cpp layer.cpp bnn.cpp mcmc.cpp -O3 -Wall
    g++-11 -o main.out main.o layer.o bnn.o mcmc.o -larmadillo
fi
if [[ "$1" == "omp" ]]
then
    g++-11 -c main.cpp layer.cpp bnn.cpp mcmc.cpp -O3 -ffast-math -march=native -fopenmp
    g++-11 -o main.out main.o layer.o bnn.o mcmc.o -larmadillo -fopenmp
fi
./main.out 