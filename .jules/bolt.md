
## 2024-05-01 - Python math.prod over np.prod for small lists
**Learning:** For small python native lists/tuples (like TensorFlow shapes or bounds limits of domains), Python's `math.prod` is significantly faster than `np.prod` because it bypasses the conversion overhead of creating an intermediate NumPy array.
**Action:** Replace `np.prod` with `math.prod` when multiplying values in list comprehensions and generators over native Python types like shape tuples.

## 2024-05-01 - TensorFlow Graph vs Eager Addition
**Learning:** Using Python's built-in `sum()` with a list comprehension of TensorFlow tensors generates an extremely inefficient execution graph containing a deep tree of sequential addition operations. Replacing it with `tf.add_n()` collapses the reduction into a single operation, resulting in significant execution time speedup.
**Action:** Use `tf.add_n()` when computing the sum of a list of TensorFlow tensors, ensuring a fallback for empty lists like `if list else tf.constant(0.0)`.
