import tensorflow as tf
import time
from src.networks.Theta import Theta

values = [tf.random.normal((100, 100)) for _ in range(50)]
other_values = [tf.random.normal((100, 100)) for _ in range(50)]
theta1 = Theta(values)
theta2 = Theta(other_values)

t0 = time.time()
for _ in range(100):
    _ = theta1 + theta2
t1 = time.time()

t2 = time.time()
for _ in range(100):
    _ = theta1 * theta2
t3 = time.time()

t4 = time.time()
for _ in range(100):
    _ = theta1 ** 2
t5 = time.time()

print(f"theta1 + theta2: {t1-t0:.4f} s")
print(f"theta1 * theta2: {t3-t2:.4f} s")
print(f"theta1 ** 2: {t5-t4:.4f} s")
