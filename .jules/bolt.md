
## 2024-05-18 - [Optimizing TensorFlow and Python operations]
**Learning:** `math.prod` is significantly faster than `np.prod` for small tuples/lists due to avoiding NumPy casting overhead. In TensorFlow, using `tf.add_n([tf.reduce_sum(tf.square(t)) ...])` is faster and avoids undefined gradients near zero compared to Python's `sum()` with `tf.norm(t)**2`.
**Action:** When calculating sizes or products of small simple lists/tuples (like tensor shapes), prefer Python's built-in `math.prod`. For summing squared values of a list of TensorFlow tensors, use `tf.add_n` and `tf.reduce_sum(tf.square())`.
