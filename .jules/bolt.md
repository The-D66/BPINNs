
## 2024-05-18 - [Optimization of Tensor Summaries and Shape Calculations]
**Learning:** In TensorFlow, operations over lists of tensors can be significantly sped up. Using `tf.add_n([tf.reduce_sum(tf.square(t)) for t in list])` is computationally more efficient than `sum([tf.norm(t)**2 for t in list])` because it avoids taking unnecessary square roots and builds a single graph node for addition. Also, for small tuples like tensor shapes, `math.prod` avoids the overhead of converting to NumPy arrays that `np.prod` introduces.
**Action:** Always prefer `tf.reduce_sum(tf.square(...))` over `tf.norm(...)**2`. For adding multiple scalar tensors in a list, utilize `tf.add_n(...)`. Use `math.prod` for multiplying elements of small tuples/lists (e.g., shape dimensions) instead of `np.prod`.
