## 2024-05-24 - tf.add_n + tf.reduce_sum(tf.square) is faster than tf.norm()**2
**Learning:** In TensorFlow, `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` is significantly faster (3.9x in benchmarks) than `sum([tf.norm(t)**2 for t in self.values])`. This is because it avoids Python's `sum()` intermediate graph bloat, and bypasses the unnecessary internal square root / squaring operations of `tf.norm()**2`, which can also cause gradient issues near zero.
**Action:** Use `tf.add_n` when aggregating loss or norms across multiple tensors instead of Python's `sum`, and prefer `tf.reduce_sum(tf.square(x))` over `tf.norm(x)**2`. Provide a type-safe fallback like `tf.constant(0.0)` for `tf.add_n` when lists could be empty.

## 2024-05-24 - math.prod is faster than np.prod for small iterables
**Learning:** When calculating the product of elements in small, simple iterables like `TensorShape` objects, `math.prod` combined with a generator expression avoids NumPy array conversion overhead and intermediate list instantiation.
**Action:** Use `sum(math.prod(shape) for shape in shapes)` instead of `sum([np.prod(shape) for shape in shapes])`.

## 2024-05-25 - tf.split vs tf.unstack(tf.expand_dims) for tensor unpacking
**Learning:** When converting a tensor into a list of its component columns (e.g., shape `(..., D)` to a list of `D` tensors of shape `(..., 1)`), using `tf.split` along the last axis is significantly faster and cleaner than doing `tf.unstack(tf.expand_dims(tensor, axis=-2), axis=-1)`. The latter involves unnecessary graph dimension manipulation. Similarly, reconstructing the tensor with `tf.concat` is faster than `tf.squeeze(tf.stack(...))`.
**Action:** Always prefer `tf.split` for slicing along an axis and `tf.concat` for joining, avoiding convoluted dimension expansions and squeezing operations where possible.
