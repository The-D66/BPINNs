## 2024-05-24 - tf.add_n + tf.reduce_sum(tf.square) is faster than tf.norm()**2
**Learning:** In TensorFlow, `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` is significantly faster (3.9x in benchmarks) than `sum([tf.norm(t)**2 for t in self.values])`. This is because it avoids Python's `sum()` intermediate graph bloat, and bypasses the unnecessary internal square root / squaring operations of `tf.norm()**2`, which can also cause gradient issues near zero.
**Action:** Use `tf.add_n` when aggregating loss or norms across multiple tensors instead of Python's `sum`, and prefer `tf.reduce_sum(tf.square(x))` over `tf.norm(x)**2`. Provide a type-safe fallback like `tf.constant(0.0)` for `tf.add_n` when lists could be empty.

## 2024-05-24 - math.prod is faster than np.prod for small iterables
**Learning:** When calculating the product of elements in small, simple iterables like `TensorShape` objects, `math.prod` combined with a generator expression avoids NumPy array conversion overhead and intermediate list instantiation.
**Action:** Use `sum(math.prod(shape) for shape in shapes)` instead of `sum([np.prod(shape) for shape in shapes])`.
## 2024-05-24 - HMC Leapfrog Gradient Caching
**Learning:** In Hamiltonian Monte Carlo (HMC) leapfrog integration, the gradient of the loss is traditionally computed at the start and end of each leapfrog step. However, the gradient evaluated at the end of step `i` is identical to the gradient evaluated at the start of step `i+1`. This redundancy causes 2L gradient evaluations per sample.
**Action:** By caching the gradient from the end of the previous step and passing it into the next step, gradient evaluations can be reduced from 2L to L+1, significantly speeding up sampling without altering mathematical correctness.
