## 2024-05-24 - tf.add_n + tf.reduce_sum(tf.square) is faster than tf.norm()**2
**Learning:** In TensorFlow, `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` is significantly faster (3.9x in benchmarks) than `sum([tf.norm(t)**2 for t in self.values])`. This is because it avoids Python's `sum()` intermediate graph bloat, and bypasses the unnecessary internal square root / squaring operations of `tf.norm()**2`, which can also cause gradient issues near zero.
**Action:** Use `tf.add_n` when aggregating loss or norms across multiple tensors instead of Python's `sum`, and prefer `tf.reduce_sum(tf.square(x))` over `tf.norm(x)**2`. Provide a type-safe fallback like `tf.constant(0.0)` for `tf.add_n` when lists could be empty.

## 2024-05-24 - math.prod is faster than np.prod for small iterables
**Learning:** When calculating the product of elements in small, simple iterables like `TensorShape` objects, `math.prod` combined with a generator expression avoids NumPy array conversion overhead and intermediate list instantiation.
**Action:** Use `sum(math.prod(shape) for shape in shapes)` instead of `sum([np.prod(shape) for shape in shapes])`.

## 2024-05-18 - [Optimize TensorFlow Unpack/Pack Operations]
**Learning:** Re-implementing unpacking a tensor along its last axis with `tf.split` is ~2.1x faster than using `tf.unstack(tf.expand_dims(tensor, axis=-2), axis=-1)`. Similarly, packing a list of tensors back together along the last axis is significantly faster with `tf.concat` compared to `tf.squeeze(tf.stack(tensor_list, axis=-1), axis=-2)`. These combined operations execute more efficiently by avoiding unnecessary intermediate tensor allocations and dimension tracking.
**Action:** Always prefer native dimensional operations like `tf.split` and `tf.concat` over composite shape manipulation tricks (`expand_dims`/`unstack` or `stack`/`squeeze`) for tensor packing and unpacking.
