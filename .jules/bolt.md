## 2024-05-06 - [TensorFlow sum over tensor operations vs tf.add_n]
**Learning:** Using Python built-in `sum([tf.norm(t)**2 for t in self.values])` to compute the squared sum of tensors is significantly slower (~3.5x slower) than using TensorFlow specific operations `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])`.
**Action:** Replace Python sum with `tf.add_n` when accumulating values across a list of tensors for performance optimization.

## 2024-05-06 - [math.prod vs np.prod for small shapes]
**Learning:** Using Python built-in `math.prod(t.shape)` to compute the size of a tensor shape is significantly faster (~8x faster) than using `np.prod(t.shape)` because of avoiding unnecessary NumPy type conversion overhead for small tuples/lists.
**Action:** Use Python`s built-in `math.prod` instead of `np.prod` when calculating the product of elements in small, simple tuples/lists (like tensor shapes or small domains).
