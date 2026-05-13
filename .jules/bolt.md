
## 2024-05-13 - [TensorFlow sum optimization]
**Learning:** Using Python's native `sum()` with `tf.norm(t)**2` when accumulating lists of tensors creates many intermediate Python objects and is significantly slower than using `tf.add_n([tf.reduce_sum(tf.square(t)) ...])`.
**Action:** When accumulating values across a list of tensors in TensorFlow, always prefer native `tf.add_n()` over Python's `sum()`. Ensure to add a fallback for empty lists when using `tf.add_n()`.
