## 2024-05-26 - TensorFlow Squared Sum Optimization
**Learning:** Using `sum([tf.norm(t)**2 for t in self.values])` to calculate the squared sum of tensors is less efficient and can lead to undefined gradients near zero because `tf.norm(t)**2` computes the square root only to square it again. Also, Python's `sum()` creates intermediate nodes in TensorFlow graphs.
**Action:** Use `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` instead of Python's `sum()` and `tf.norm(t)**2`.

## 2024-05-26 - Python built-in math.prod vs np.prod
**Learning:** `np.prod` has high overhead for simple iterables or lists/tuples of integers due to array coercion. `math.prod` is implemented in C and runs much faster on simple Python lists/tuples.
**Action:** Replace `np.prod` with `math.prod` for calculating the product of elements in simple shapes/domains (like `np.prod(t.shape)`).
