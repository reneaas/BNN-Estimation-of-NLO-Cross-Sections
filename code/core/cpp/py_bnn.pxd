
from libcpp.string cimport string
from libcpp.vector cimport vector

from py_layer cimport Layer

cdef extern from "bnn.cpp":
    pass

cdef extern from "bnn.hpp":
    cdef cppclass BNN:
        BNN() except +
        BNN(vector[int] layers, string activation) except +

        void set_weights(vector[vector[double]] weights)
        vector[vector[double]] get_weights()
        vector[vector[double]] get_gradients()

        vector[double] forward(vector[double] x)
        void backward(vector[double] x, vector[double] y)
        void mle_fit(
            vector[vector[double]] X,
            vector[vector[double]] Y,
            int num_epochs,
            double lr,
        )

    