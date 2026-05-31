## 2024-05-24 - Tensor Reduction Optimization
**Learning:** In TensorFlow, `sum([tf.norm(t)**2 for t in values])` is significantly slower and less numerically stable than `tf.add_n([tf.reduce_sum(tf.square(t)) for t in values])`. Furthermore, using Python's `sum()` to add a list of tensors creates an inefficient, deep computational graph compared to using TensorFlow's native `tf.add_n()`.
**Action:** Always prefer `tf.add_n()` for accumulating lists of tensors, and avoid taking square roots (`tf.norm`) just to square them again. Always include an empty list check with a `tf.constant(0.0)` fallback when using `tf.add_n()` on dynamically sized lists (like loss dictionaries).

## 2024-05-24 - NumPy vs Math Built-ins
**Learning:** `np.prod` has significant overhead compared to Python's built-in `math.prod` when operating on small lists/tuples, like tensor shape dimensions or small coordinate bounding boxes.
**Action:** Use `math.prod` instead of `np.prod` for small collections to avoid NumPy type conversion overhead.
