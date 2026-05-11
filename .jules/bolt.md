## 2026-05-11 - [TensorFlow sum optimization]
**Learning:** Python's built-in `sum()` function is inefficient when working with a list of TensorFlow tensors, leading to numerous intermediate operations. Furthermore, `tf.norm(t)**2` is slower and can lead to mathematically unstable gradients near zero.
**Action:** When summing squares of tensors, use `tf.add_n([tf.reduce_sum(tf.square(t)) for t in ...])`. Always include a fallback for empty lists like `if not self.values: return tf.constant(0.0, dtype=tf.float32)` since `tf.add_n` throws an error on empty lists.

## 2026-05-11 - [Numpy prod optimization]
**Learning:** `np.prod` has a significant overhead when applied to small, simple tuples (like tensor shapes) due to type conversion.
**Action:** Use Python's built-in `math.prod` instead of `np.prod` for calculating the product of elements in small, simple tuples or lists (like tensor shapes) to avoid unnecessary overhead.
