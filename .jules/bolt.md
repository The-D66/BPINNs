## 2024-05-24 - tf.add_n + tf.reduce_sum(tf.square) is faster than tf.norm()**2
**Learning:** In TensorFlow, `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` is significantly faster (3.9x in benchmarks) than `sum([tf.norm(t)**2 for t in self.values])`. This is because it avoids Python's `sum()` intermediate graph bloat, and bypasses the unnecessary internal square root / squaring operations of `tf.norm()**2`, which can also cause gradient issues near zero.
**Action:** Use `tf.add_n` when aggregating loss or norms across multiple tensors instead of Python's `sum`, and prefer `tf.reduce_sum(tf.square(x))` over `tf.norm(x)**2`. Provide a type-safe fallback like `tf.constant(0.0)` for `tf.add_n` when lists could be empty.

## 2024-05-24 - math.prod is faster than np.prod for small iterables
**Learning:** When calculating the product of elements in small, simple iterables like `TensorShape` objects, `math.prod` combined with a generator expression avoids NumPy array conversion overhead and intermediate list instantiation.
**Action:** Use `sum(math.prod(shape) for shape in shapes)` instead of `sum([np.prod(shape) for shape in shapes])`.
## 2024-05-24 - tf.split vs tf.unstack and tf.concat vs tf.stack for Lists of Tensors
**Learning:** When unpacking a tensor into a list of its slices along the last dimension, `tf.split(tensor, tensor.shape[-1], axis=-1)` avoids intermediate dimension additions/manipulations compared to `tf.unstack(tf.expand_dims(tensor, axis=-2), axis=-1)` and executes ~38% faster in eager mode. Similarly, packing them back using `tf.concat(tensor_list, axis=-1)` is noticeably faster than `tf.squeeze(tf.stack(tensor_list, axis=-1), axis=-2)`.
**Action:** Use `tf.split` and `tf.concat` when dealing with extracting or reconstructing continuous tensor slices, avoiding expensive intermediate dimension expansion or stacking/squeezing.
