## 2024-05-24 - tf.add_n + tf.reduce_sum(tf.square) is faster than tf.norm()**2
**Learning:** In TensorFlow, `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` is significantly faster (3.9x in benchmarks) than `sum([tf.norm(t)**2 for t in self.values])`. This is because it avoids Python's `sum()` intermediate graph bloat, and bypasses the unnecessary internal square root / squaring operations of `tf.norm()**2`, which can also cause gradient issues near zero.
**Action:** Use `tf.add_n` when aggregating loss or norms across multiple tensors instead of Python's `sum`, and prefer `tf.reduce_sum(tf.square(x))` over `tf.norm(x)**2`. Provide a type-safe fallback like `tf.constant(0.0)` for `tf.add_n` when lists could be empty.

## 2024-05-24 - math.prod is faster than np.prod for small iterables
**Learning:** When calculating the product of elements in small, simple iterables like `TensorShape` objects, `math.prod` combined with a generator expression avoids NumPy array conversion overhead and intermediate list instantiation.
**Action:** Use `sum(math.prod(shape) for shape in shapes)` instead of `sum([np.prod(shape) for shape in shapes])`.

## 2024-05-25 - tf.split vs tf.unstack / tf.expand_dims
**Learning:** `tf.split(tensor, tensor.shape[-1], axis=-1)` is significantly faster (approx. 35% faster in graph mode benchmarks) than `tf.unstack(tf.expand_dims(tensor, axis=-2), axis=-1)` and avoids unnecessary dimension manipulation when breaking a tensor into a list of column tensors.
**Action:** Use `tf.split` instead of `tf.unstack` + `tf.expand_dims` when extracting column tensors.

## 2024-05-25 - tf.concat vs tf.squeeze / tf.stack
**Learning:** `tf.concat(tensor_list, axis=-1)` is simpler and marginally faster than `tf.squeeze(tf.stack(tensor_list, axis=-1), axis=-2)` when packing a list of column tensors back into a single tensor, as it avoids creating and immediately removing a dimension.
**Action:** Use `tf.concat` directly on lists of appropriately shaped tensors instead of stacking and squeezing.
