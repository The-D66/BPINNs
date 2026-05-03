## 2025-05-24 - Efficient List Accumulation in TensorFlow
**Learning:** Using Python's built-in `sum()` function to accumulate values across a list of tensors creates many intermediate Python objects and is significantly slower than using native TensorFlow operators.
**Action:** Use `tf.add_n()` instead of Python's `sum()` when accumulating values across a list of tensors (e.g., in `tf_trace`). Always remember to add a check for an empty list since `tf.add_n` will crash if passed an empty list, whereas `sum()` returns 0.
