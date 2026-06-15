## 2024-05-24 - tf.add_n + tf.reduce_sum(tf.square) is faster than tf.norm()**2
**Learning:** In TensorFlow, `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` is significantly faster (3.9x in benchmarks) than `sum([tf.norm(t)**2 for t in self.values])`. This is because it avoids Python's `sum()` intermediate graph bloat, and bypasses the unnecessary internal square root / squaring operations of `tf.norm()**2`, which can also cause gradient issues near zero.
**Action:** Use `tf.add_n` when aggregating loss or norms across multiple tensors instead of Python's `sum`, and prefer `tf.reduce_sum(tf.square(x))` over `tf.norm(x)**2`. Provide a type-safe fallback like `tf.constant(0.0)` for `tf.add_n` when lists could be empty.

## 2024-05-24 - math.prod is faster than np.prod for small iterables
**Learning:** When calculating the product of elements in small, simple iterables like `TensorShape` objects, `math.prod` combined with a generator expression avoids NumPy array conversion overhead and intermediate list instantiation.
**Action:** Use `sum(math.prod(shape) for shape in shapes)` instead of `sum([np.prod(shape) for shape in shapes])`.

## 2024-05-25 - Caching grad_loss in HMC leapfrog integration
**Learning:** In Hamiltonian Monte Carlo (HMC) leapfrog integration (`src/algorithms/HMC.py`), computing `grad_loss` twice per step is redundant. The final gradient calculated at the end of step `n` is identical to the initial gradient needed at the start of step `n+1`.
**Action:** Cache the gradient calculated at the end of a leapfrog step and pass it as an argument to the beginning of the next step. This reduces gradient evaluations from 2*L to L+1 per parameter sample, yielding significant speedups during parameter sampling. Ensure code changes explain this optimization logic.
