## 2024-05-14 - TensorFlow and Math Operations Micro-Optimizations
**Learning:**
1. `tf.norm(t)**2` combined with Python's built-in `sum()` is significantly slower than `tf.add_n([tf.reduce_sum(tf.square(t)) ...])`. The former computes unnecessary square roots and creates a deep Python addition graph. The latter uses C++ native tensor addition and avoids the square root overhead, providing up to a 4x speedup in eager mode and almost 2x when compiled with `@tf.function`.
2. Python's `math.prod` is significantly faster (~10x) than `np.prod` for multiplying elements in small, simple tuples or lists (like tensor shapes `t.shape` or small domain bound arrays) because it avoids the overhead of converting iterables to NumPy arrays.
3. `tf.add_n` versus `sum()` in other contexts, like inside `tf_trace`, provides marginal to zero speedup when compiled, so it's not universally beneficial to swap everywhere unless it's a known bottleneck like L2 norm accumulation.
**Action:**
- Always replace `sum([tf.norm(t)**2 ...])` with `tf.add_n([tf.reduce_sum(tf.square(t)) ...])` with a fallback condition for empty lists.
- Use `math.prod` instead of `np.prod` for small integer/float iterables that are not already NumPy arrays.
