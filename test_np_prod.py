import tensorflow as tf
import time
import numpy as np
from src.networks.Theta import Theta

values = [tf.random.normal((100, 100)) for _ in range(50)]
theta = Theta(values)

# Warm up
_ = sum([np.prod(t.shape) for t in theta.values])
_ = sum([tf.size(t).numpy() for t in theta.values])

t0 = time.time()
for _ in range(100):
    _ = sum([np.prod(t.shape) for t in theta.values])
t1 = time.time()

t2 = time.time()
for _ in range(100):
    _ = sum([tf.size(t).numpy() for t in theta.values])
t3 = time.time()

t4 = time.time()
for _ in range(100):
    _ = sum(tf.size(t) for t in theta.values)
t5 = time.time()

print(f"np.prod(t.shape): {t1-t0:.4f} s")
print(f"tf.size(t).numpy(): {t3-t2:.4f} s")
print(f"tf.size(t): {t5-t4:.4f} s")
