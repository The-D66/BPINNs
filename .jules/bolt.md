
## 2024-05-18 - Avoid Python `sum` for list of TensorFlow tensors
**Learning:** Using Python's built-in `sum([tf.norm(t)**2 ...])` is inefficient because it creates intermediate Python objects. Additionally, computing `tf.norm(t)**2` computes a square root and then squares it, which is numerically less stable and computationally more expensive than just `tf.reduce_sum(tf.square(t))`.
**Action:** When accumulating across lists of TensorFlow tensors, use `tf.add_n([tf.reduce_sum(tf.square(t)) for t in lst])` and always include a fallback (like `if lst else tf.constant(0.0)`) to avoid exceptions on empty lists.

## 2024-05-18 - Math.prod vs NumPy.prod for small tuples
**Learning:** Using `np.prod` on small native Python tuples (e.g., `t.shape` from TensorFlow) introduces significant overhead because NumPy first converts the tuple into a C-array.
**Action:** Use Python's built-in `math.prod` for finding the product of elements in small, simple tuples/lists (like tensor shapes).
