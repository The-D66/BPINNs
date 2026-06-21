## 2024-05-24 - tf.add_n + tf.reduce_sum(tf.square) is faster than tf.norm()**2
**Learning:** In TensorFlow, `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` is significantly faster (3.9x in benchmarks) than `sum([tf.norm(t)**2 for t in self.values])`. This is because it avoids Python's `sum()` intermediate graph bloat, and bypasses the unnecessary internal square root / squaring operations of `tf.norm()**2`, which can also cause gradient issues near zero.
**Action:** Use `tf.add_n` when aggregating loss or norms across multiple tensors instead of Python's `sum`, and prefer `tf.reduce_sum(tf.square(x))` over `tf.norm(x)**2`. Provide a type-safe fallback like `tf.constant(0.0)` for `tf.add_n` when lists could be empty.

## 2024-05-24 - math.prod is faster than np.prod for small iterables
**Learning:** When calculating the product of elements in small, simple iterables like `TensorShape` objects, `math.prod` combined with a generator expression avoids NumPy array conversion overhead and intermediate list instantiation.
**Action:** Use `sum(math.prod(shape) for shape in shapes)` instead of `sum([np.prod(shape) for shape in shapes])`.

## 2024-05-24 - HMC Leapfrog Gradient Caching
**Learning:** In Hamiltonian Monte Carlo (HMC) leapfrog integration, evaluating the gradient of the loss function is the most expensive operation. The gradient computed at the end of leapfrog step `i` is identical to the gradient needed at the beginning of leapfrog step `i+1` because the parameters (`theta`) do not change between those two points. Recomputing it is redundant.
**Action:** When implementing or optimizing leapfrog integrators, cache the gradient from the end of the previous step and pass it to the next step. This reduces gradient evaluations from $2L$ to $L+1$ per trajectory, essentially halving the computational cost of the sampling loop.
