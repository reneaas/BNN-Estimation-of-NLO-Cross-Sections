

from libcpp.string cimport string
from libcpp.vector cimport vector
from py_layer cimport Layer

cdef class PyLayer:
    cdef Layer c_layer

    def __cinit__(self, int n_rows, int n_cols, string activation):
        self.c_layer = Layer(n_rows, n_cols, activation)
    
    def get_kernel(self):
        return self.c_layer.get_kernel()

    def get_bias(self):
        return self.c_layer.get_bias()


    @property
    def bias(self):
        return self.c_layer.b_

    @property
    def kernel(self):
        return self.c_layer.w_

    @property
    def weights(self):
        return (self.c_layer.w_, self.c_layer.b_)

