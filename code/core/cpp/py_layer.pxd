
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "layer.cpp":
    pass

cdef extern from "layer.hpp":
    cdef cppclass Layer:
        Layer() except +
        Layer(int n_rows, int n_cols, string activation) except +
        vector[double] w_, b_, z_, a_
        int n_rows_, n_cols_

        vector[double] get_kernel()
        vector[double] get_bias()

    

