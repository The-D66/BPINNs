
## 2024-04-16 - [TensorFlow Python Optimization - ssum and size on Tensor tuples]
**Learning:** In a codebase manipulating tuples or lists of TensorFlow tensors or their shapes, you should avoid `np.prod` for small sets like shapes (use Python 3.8+ `math.prod`), and avoid Python's `sum()` wrapping TensorFlow methods if they can be combined inside TensorFlow via `tf.add_n` + `tf.reduce_sum(tf.square())` rather than `sum()` + `tf.norm()**2`.
**Action:** When computing element-wise statistics of an iterable of Tensors, stay strictly within native Python math functions (`math.prod` instead of `np.prod` to avoid conversions) or strict TensorFlow operations (`tf.add_n` instead of Python `sum`).
