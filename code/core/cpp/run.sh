
if [[ "$1" == "bnn" ]]
then
    g++-11 -c main.cpp layer.cpp bnn.cpp mcmc.cpp -O3 -Wall
    g++-11 -o main.out main.o layer.o bnn.o mcmc.o -larmadillo
fi
if [[ "$1" == "nn" ]]
then
    g++-11 -c main.cpp layer.cpp nn.cpp -O3 -Wall
    g++-11 -o main.out main.o layer.o nn.o
fi
./main.out 