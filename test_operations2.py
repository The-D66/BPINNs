import tensorflow as tf
import time

values = [tf.random.normal((100, 100)) for _ in range(50)]
other_values = [tf.random.normal((100, 100)) for _ in range(50)]

t0 = time.time()
for _ in range(100):
    _ = [a+b for a,b in zip(values, other_values)]
t1 = time.time()

t2 = time.time()
for _ in range(100):
    _ = tf.nest.map_structure(tf.add, values, other_values)
t3 = time.time()

t4 = time.time()
for _ in range(100):
    _ = tf.nest.map_structure(lambda a, b: a+b, values, other_values)
t5 = time.time()

print(f"[a+b for a,b in zip(...) ]: {t1-t0:.4f} s")
print(f"tf.nest.map_structure(tf.add, ...): {t3-t2:.4f} s")
print(f"tf.nest.map_structure(lambda, ...): {t5-t4:.4f} s")
