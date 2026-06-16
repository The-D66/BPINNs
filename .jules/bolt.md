## 2024-05-24 - tf.add_n + tf.reduce_sum(tf.square) is faster than tf.norm()**2
**Learning:** In TensorFlow, `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` is significantly faster (3.9x in benchmarks) than `sum([tf.norm(t)**2 for t in self.values])`. This is because it avoids Python's `sum()` intermediate graph bloat, and bypasses the unnecessary internal square root / squaring operations of `tf.norm()**2`, which can also cause gradient issues near zero.
**Action:** Use `tf.add_n` when aggregating loss or norms across multiple tensors instead of Python's `sum`, and prefer `tf.reduce_sum(tf.square(x))` over `tf.norm(x)**2`. Provide a type-safe fallback like `tf.constant(0.0)` for `tf.add_n` when lists could be empty.

## 2024-05-24 - math.prod is faster than np.prod for small iterables
**Learning:** When calculating the product of elements in small, simple iterables like `TensorShape` objects, `math.prod` combined with a generator expression avoids NumPy array conversion overhead and intermediate list instantiation.
**Action:** Use `sum(math.prod(shape) for shape in shapes)` instead of `sum([np.prod(shape) for shape in shapes])`.

## 2024-05-24 - HMC Leapfrog Gradient Caching
**Learning:** In Hamiltonian Monte Carlo (HMC) leapfrog integration, evaluating the model gradient (`grad_loss`) at the end of leapfrog step `i` and then again at the beginning of step `i+1` is redundant. Modifying `__leapfrog_step` to return the calculated `grad_theta` and passing it sequentially from one step to the next reduces total gradient evaluations from `2*HMC_L` to `HMC_L + 1` per sample, resulting in almost a 2x speedup during sampling.
**Action:** When implementing or optimizing HMC or similar iterative update loops (like leapfrog integrators) where the state transitions sequentially, cache and reuse expensive intermediate evaluations (like gradients or loss functions) between loop iterations to avoid redundant computation.
