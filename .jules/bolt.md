## 2024-05-14 - Replace Python's `sum()` and `tf.norm(t)**2` with `tf.add_n` and `tf.reduce_sum(tf.square(t))`
**Learning:** In TensorFlow operations over a list of tensors, using Python's built-in `sum()` with `tf.norm(t)**2` is significantly slower than using native TensorFlow operators. `tf.add_n` combined with `tf.reduce_sum(tf.square(t))` avoids the overhead of creating many intermediate Python objects and leverages optimized native C++ ops for accumulating sums across multiple tensors.
**Action:** When accumulating values across a list of TensorFlow tensors, always prefer using `tf.add_n` over Python's built-in `sum()`. For squared norm calculations, use `tf.reduce_sum(tf.square(t))` instead of `tf.norm(t)**2`.

## 2024-05-14 - Replace `np.prod` with `math.prod` for small tuples like Tensor shapes
**Learning:** `np.prod` converts its input into a NumPy array, which introduces significant overhead when the input is just a small Python tuple (like a Tensor shape `(100, 100)`). The standard library `math.prod` calculates the product directly on the Python objects, avoiding this costly conversion.
**Action:** When calculating the product of elements in small Python collections (such as tensor shapes), use `math.prod` from the standard library instead of `np.prod`.

## 2024-05-14 - Replace Python's `sum()` with `tf.add_n` in `tf_trace`
**Learning:** Even for accumulating slices of a tensor in operations like `tf_trace` where you map over an index `sum([v[:,i] for i, v in enumerate(lt)])`, using TensorFlow's native `tf.add_n` is measurably faster than python's built-in `sum()`.
**Action:** Replace `sum()` with `tf.add_n` for accumulating lists of tensors everywhere in the codebase.
