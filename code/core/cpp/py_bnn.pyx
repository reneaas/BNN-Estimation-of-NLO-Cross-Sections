
from libcpp.string cimport string
from libcpp.vector cimport vector
from py_bnn cimport BNN

cdef class PyBNN:
    cdef BNN c_bnn
    
    def __cinit__(self, vector[int] layers, string activation):
        self.c_bnn = BNN(layers, activation)
    
    def get_weights(self):
        return self.c_bnn.get_weights()

    def forward(self, vector[double] x):
        return self.c_bnn.forward(x)

    def backward(self, vector[double] x, vector[double] y):
        return self.c_bnn.backward(x, y)
        
    def mle_fit(self, 
        vector[vector[double]] X, 
        vector[vector[double]] Y,
        int num_epochs, 
        double lr,
    ):
        return self.c_bnn.mle_fit(X, Y, num_epochs, lr)

    @property
    def weights(self):
        return self.c_bnn.get_weights()

    @weights.setter
    def weights(self, vector[vector[double]] weights):
        self.c_bnn.set_weights(weights)

    @property
    def gradients(self):
        return self.c_bnn.get_gradients()
    

    
