import tensorflow as tf
import time
import math
from src.networks.Theta import Theta

values = [tf.random.normal((100, 100)) for _ in range(50)]

# Original size
def size_original(values):
    import numpy as np
    return sum([np.prod(t.shape) for t in values])

# Optimized size
def size_optimized(values):
    import math
    return sum(math.prod(t.shape) for t in values)

print(f"Original: {size_original(values)}")
print(f"Optimized: {size_optimized(values)}")

t0 = time.time()
for _ in range(100):
    _ = size_original(values)
t1 = time.time()

t2 = time.time()
for _ in range(100):
    _ = size_optimized(values)
t3 = time.time()

print(f"Original: {t1-t0:.4f} s")
print(f"Optimized: {t3-t2:.4f} s")
