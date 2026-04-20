## 2024-05-19 - TensorFlow Operation Optimization
**Learning:** Python's built-in `sum()` with `tf.norm(t)**2` is extremely slow when summing lists of TensorFlow tensors compared to using `tf.add_n([tf.reduce_sum(tf.square(t)) ...])`. Furthermore, `np.prod` introduces significant overhead compared to Python's built-in `math.prod` for small tuples like tensor shapes.
**Action:** When calculating squared sums of tensors, use `tf.add_n` and `tf.reduce_sum`. When calculating the product of elements in small, simple tuples, prefer `math.prod` over `np.prod` to avoid numpy type conversion overhead.
