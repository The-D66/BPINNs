## 2024-05-24 - tf.add_n + tf.reduce_sum(tf.square) is faster than tf.norm()**2
**Learning:** In TensorFlow, `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` is significantly faster (3.9x in benchmarks) than `sum([tf.norm(t)**2 for t in self.values])`. This is because it avoids Python's `sum()` intermediate graph bloat, and bypasses the unnecessary internal square root / squaring operations of `tf.norm()**2`, which can also cause gradient issues near zero.
**Action:** Use `tf.add_n` when aggregating loss or norms across multiple tensors instead of Python's `sum`, and prefer `tf.reduce_sum(tf.square(x))` over `tf.norm(x)**2`. Provide a type-safe fallback like `tf.constant(0.0)` for `tf.add_n` when lists could be empty.

## 2024-05-24 - math.prod is faster than np.prod for small iterables
**Learning:** When calculating the product of elements in small, simple iterables like `TensorShape` objects, `math.prod` combined with a generator expression avoids NumPy array conversion overhead and intermediate list instantiation.
**Action:** Use `sum(math.prod(shape) for shape in shapes)` instead of `sum([np.prod(shape) for shape in shapes])`.

## 2026-06-08 - tf.add_n vs tf.linalg.trace memory/speed trade-offs
**Learning:** When calculating traces of Jacobians or matrices (lists of column vectors) during loss computation, stacking tensors into a matrix to use `tf.linalg.trace(tf.stack(lt, axis=-1))` causes huge memory spikes and slowdowns compared to `tf.expand_dims(sum([v[:,i] for i, v in enumerate(lt)]), axis=-1)`. However, replacing Python's `sum()` with TensorFlow's native `tf.add_n()` avoids intermediate object allocations while offering slight speedups in both eager and compiled modes over Python's `sum()`.
**Action:** When computing sums of tensors (like traces from lists of columns or accumulating dictionary losses), use `tf.add_n(list_of_tensors)` rather than `sum()` to avoid memory/graph bloat, but do not refactor into `tf.stack` + `tf.linalg.trace`. Provide a fallback `tf.constant(0.0, dtype=tf.float32)` for `tf.add_n` when lists could be empty.
