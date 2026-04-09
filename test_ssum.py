import tensorflow as tf
import time
from src.networks.Theta import Theta

# Create some dummy values
values = [tf.random.normal((100, 100)) for _ in range(50)]
theta = Theta(values)

t0 = time.time()
for _ in range(100):
    _ = sum([tf.norm(t)**2 for t in theta.values])
t1 = time.time()

t2 = time.time()
for _ in range(100):
    _ = sum([tf.reduce_sum(tf.square(t)) for t in theta.values])
t3 = time.time()

t4 = time.time()
for _ in range(100):
    _ = sum([tf.reduce_sum(t**2) for t in theta.values])
t5 = time.time()


t6 = time.time()
for _ in range(100):
    _ = tf.add_n([tf.reduce_sum(tf.square(t)) for t in theta.values])
t7 = time.time()

print(f"tf.norm(t)**2: {t1-t0:.4f} s")
print(f"sum tf.reduce_sum(tf.square(t)): {t3-t2:.4f} s")
print(f"sum tf.reduce_sum(t**2): {t5-t4:.4f} s")
print(f"tf.add_n([tf.reduce_sum(tf.square(t))]): {t7-t6:.4f} s")
