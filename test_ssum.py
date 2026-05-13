import tensorflow as tf
import numpy as np
import time

# Create a dummy Theta-like class for testing
class DummyTheta:
    def __init__(self, values):
        self.values = values

    def ssum_original(self):
        return sum([tf.norm(t)**2 for t in self.values])

    def ssum_optimized(self):
        if not self.values:
            return tf.constant(0.0)
        return tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])

# Create list of tensors
tf.random.set_seed(42)
tensors = [tf.random.normal((100, 100)) for _ in range(50)]
theta = DummyTheta(tensors)

# Warmup
theta.ssum_original()
theta.ssum_optimized()

# Benchmark
import timeit

def benchmark(func, name, n=1000):
    start = time.time()
    for _ in range(n):
        func()
    end = time.time()
    print(f"{name}: {end - start:.4f} seconds")

benchmark(theta.ssum_original, "Original sum() + tf.norm()**2")
benchmark(theta.ssum_optimized, "Optimized tf.add_n() + tf.reduce_sum(tf.square())")
