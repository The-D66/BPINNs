import tensorflow as tf
import time
from src.networks.Theta import Theta

values = [tf.random.normal((100, 100)) for _ in range(50)]

# Original ssum
def ssum_original(values):
    return sum([tf.norm(t)**2 for t in values])

# Optimized ssum
def ssum_optimized(values):
    return tf.add_n([tf.reduce_sum(tf.square(t)) for t in values])

# Warm up
_ = ssum_original(values)
_ = ssum_optimized(values)

print(f"Difference: {tf.abs(ssum_original(values) - ssum_optimized(values)):.6f}")

t0 = time.time()
for _ in range(100):
    _ = ssum_original(values)
t1 = time.time()

t2 = time.time()
for _ in range(100):
    _ = ssum_optimized(values)
t3 = time.time()

print(f"Original: {t1-t0:.4f} s")
print(f"Optimized: {t3-t2:.4f} s")
