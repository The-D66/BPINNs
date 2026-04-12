import time
import math
import numpy as np
import tensorflow as tf

# Mock the setup
tensors = [tf.random.normal((500, 500)) for _ in range(10)] + [tf.random.normal((500,)) for _ in range(10)]

def ssum_old():
    return sum([tf.norm(t)**2 for t in tensors])

def ssum_new():
    return tf.add_n([tf.reduce_sum(tf.square(t)) for t in tensors])

def size_old():
    return sum([np.prod(t.shape) for t in tensors])

def size_new():
    return sum([math.prod(t.shape) for t in tensors])

# Warmup
ssum_old()
ssum_new()
size_old()
size_new()

import timeit

print("ssum_old:", timeit.timeit(ssum_old, number=1000))
print("ssum_new:", timeit.timeit(ssum_new, number=1000))
print("size_old:", timeit.timeit(size_old, number=10000))
print("size_new:", timeit.timeit(size_new, number=10000))
