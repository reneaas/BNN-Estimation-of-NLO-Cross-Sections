from integrate import midpoint
from integrate import matmul
import numpy as np
from time import time

a = 0
b = 1
n = 1000000000
integral = midpoint(a, b, n)
print(integral)

n = 1500

A = np.random.normal(size=(n,n))
B = np.random.normal(size=(n,n))
start = time()
C = np.array(matmul(A, B, n))
end = time()
timeused = end-start
print(timeused)
print("----"*20)
start = time()
D = np.matmul(A,B)
end= time()
timeused = end-start
print(timeused)
