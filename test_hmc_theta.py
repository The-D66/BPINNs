import tensorflow as tf
import time
from src.networks.Theta import Theta

values = [tf.random.normal((100, 100)) for _ in range(50)]
theta = Theta(values)

t0 = time.time()
for _ in range(100):
    _ = theta.normal(1.0)
t1 = time.time()

t2 = time.time()
for _ in range(100):
    _ = Theta([tf.random.normal(t.shape, mean=1.0, stddev=1.0) for t in theta.values])
t3 = time.time()

print(f"theta.normal(): {t1-t0:.4f} s")
print(f"Theta([...]): {t3-t2:.4f} s")
