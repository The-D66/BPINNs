## 2024-05-24 - tf.add_n + tf.reduce_sum(tf.square) is faster than tf.norm()**2
**Learning:** In TensorFlow, `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` is significantly faster (3.9x in benchmarks) than `sum([tf.norm(t)**2 for t in self.values])`. This is because it avoids Python's `sum()` intermediate graph bloat, and bypasses the unnecessary internal square root / squaring operations of `tf.norm()**2`, which can also cause gradient issues near zero.
**Action:** Use `tf.add_n` when aggregating loss or norms across multiple tensors instead of Python's `sum`, and prefer `tf.reduce_sum(tf.square(x))` over `tf.norm(x)**2`. Provide a type-safe fallback like `tf.constant(0.0)` for `tf.add_n` when lists could be empty.

## 2024-05-24 - math.prod is faster than np.prod for small iterables
**Learning:** When calculating the product of elements in small, simple iterables like `TensorShape` objects, `math.prod` combined with a generator expression avoids NumPy array conversion overhead and intermediate list instantiation.
**Action:** Use `sum(math.prod(shape) for shape in shapes)` instead of `sum([np.prod(shape) for shape in shapes])`.
## 2026-06-28 - tf.add_n is faster than Python's sum() for lists of tensors
**Learning:** In TensorFlow, applying Python's `sum()` function to a list of tensors creates -1$ individual `Add` operations in the computation graph, which increases compilation time and memory overhead. Using `tf.add_n()` instead creates a single N-ary `AddN` operation, significantly reducing graph bloat and execution time (around 6x faster in a simple trace benchmark). However, it requires a fallback for empty lists as it throws a `ValueError`.
**Action:** Always replace `sum(list_of_tensors)` with `tf.add_n(list_of_tensors) if list_of_tensors else tf.constant(0.0)` when aggregating loss components or traces.
