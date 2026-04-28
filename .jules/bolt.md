## 2024-04-28 - Optimize TensorFlow Tensor Summarization
**Learning:** Using Python's built-in `sum()` function with `tf.norm(t)**2` to accumulate squared values over a list of TensorFlow tensors is inefficient and can cause undefined gradients near zero (due to the square root in the norm).
**Action:** Replace `sum([tf.norm(t)**2 for t in tensors])` with `tf.add_n([tf.reduce_sum(tf.square(t)) for t in tensors])`. This uses native, highly optimized TensorFlow operations and handles zero values cleanly, improving speed significantly (e.g., ~3.5x faster in some tests).

## 2024-04-28 - Optimize Small Shape Products
**Learning:** Using `np.prod()` to calculate the product of small lists/tuples (like tensor shapes or small domain bounds) incurs unnecessary type conversion overhead in Python.
**Action:** Use Python's built-in `math.prod()` instead of `np.prod()` for these small, native data structures. This optimization avoids NumPy's overhead and executes roughly 10x faster.
