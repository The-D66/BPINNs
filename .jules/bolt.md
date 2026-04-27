## 2026-04-27 - [Optimize Theta metric computations]
**Learning:** Using Python's `sum` with a list comprehension of `tf.norm(t)**2` results in excessive TensorFlow graph nodes and overhead, making it drastically slower than vectorized native TF operations.
**Action:** Use `tf.add_n([tf.reduce_sum(tf.square(t)) ...])` to efficiently compute the squared norm sum over a list of tensors. Additionally, when compiling size of structures, prefer Python's `math.prod` over `np.prod` to avoid casting overheads on small iterables like tensor shapes.
