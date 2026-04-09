import tensorflow as tf
import time
from src.networks.Theta import Theta

# Create some dummy values
values = [tf.random.normal((100, 100)) for _ in range(50)]
theta = Theta(values)

t0 = time.time()
for _ in range(100):
    _ = theta.ssum()
t1 = time.time()

print(f"theta.ssum(): {t1-t0:.4f} s")
