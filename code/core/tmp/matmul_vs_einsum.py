import tensorflow as tf
import time



n = 10000
m = 5000
x = tf.random.normal(shape=(n, m))
w = tf.random.normal(shape=(m, m))


start = time.perf_counter()
res = tf.einsum("...j,ij->...i", x, w)
end = time.perf_counter()
timeused = end - start
print(f"{timeused=} seconds with einsum")


start = time.perf_counter()
res = tf.matmul(x, w)
end = time.perf_counter()
timeused = end - start
print(f"{timeused=} seconds with matmul")