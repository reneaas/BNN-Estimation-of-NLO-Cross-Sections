import numpy as np
cimport numpy as np

from cython.parallel import prange, parallel
import cython

cdef double f(double x) nogil:
    cdef double res = x*x*x
    return res

def midpoint(double a, double b, int n):

    cdef int i
    cdef double I = 0.
    cdef double h
    cdef double x
    h = (b-a)*(1./n)

    for i in prange(n, nogil=True):
        x = a + i*h
        I += f(x)

    return I*h


@cython.boundscheck(False)
@cython.wraparound(False)
def matmul(double [:, :] A, double [:, :] B, int n):

    cdef int i, j, k
    cdef double [:, :] C = np.zeros((n,n))
    for i in prange(n, nogil=True):
        for j in range(n):
            for k in range(n):
                C[i,j] += A[i,k]*B[k,j]
    return C
