all: compile link run

compile:
	g++-11 -c *.cpp -O3 -fopenmp -std=c++17

link:
	g++-11 -o main.out *.o -fopenmp -larmadillo

run:
	./main.out
