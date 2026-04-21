## 2024-05-19 - Fast Tensor Reductions and Shape Products
**Learning:** In TensorFlow, computing a squared norm sum via `sum([tf.norm(t)**2 ...])` is significantly slower than building a single fused graph operation via `tf.add_n([tf.reduce_sum(tf.square(t)) ...])`. For tiny tensor tuples like tensor shapes, Numpy's `np.prod` adds noticeable overhead compared to Python's built-in `math.prod`.
**Action:** Use `tf.add_n` + `tf.reduce_sum` + `tf.square` for large iterative tensor aggregations and prefer `math.prod` for small integer tuples.
