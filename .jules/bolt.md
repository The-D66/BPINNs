## 2024-05-12 - TensorFlow ssum vs add_n and reduce_sum
**Learning:** Using `sum([tf.norm(t)**2 for t in values])` is significantly slower than `tf.add_n([tf.reduce_sum(tf.square(t)) for t in values])` for accumulating squared values of a list of TensorFlow tensors. The optimized method avoids computing unnecessary square roots with `tf.norm`, avoids using Python's primitive `sum()` which causes overhead, and leverages `tf.add_n` for native TensorFlow graph optimization. Always remember to add a conditional like `if values else 0.0` when using `tf.add_n` on a potentially empty list to avoid errors.

## 2024-05-12 - math.prod vs np.prod for small tensor shape multiplication
**Learning:** Using `np.prod` for small tuples like tensor shapes inside a loop incurs unnecessary NumPy type conversion overhead. Python's built-in `math.prod` is significantly faster for these operations. Using a generator expression `sum(math.prod(...) ...)` instead of a list comprehension also avoids building an intermediate list.
**Action:** Prefer `math.prod` with a generator expression over `np.prod` with list comprehension for operations dealing with small simple types or tuples like tensor shapes.
