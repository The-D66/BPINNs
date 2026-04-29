## 2025-02-25 - [TensorFlow Accumulation in tf_trace]
**Learning:** Python's built-in `sum()` function is a major performance bottleneck when used to sum lists of TensorFlow tensors because it builds an unbalanced and deep computation graph (`(((a+b)+c)+d)...`), leading to high Python and execution overhead.
**Action:** Always use `tf.add_n()` to accumulate values across lists of tensors to ensure elementwise addition occurs efficiently as a single C++ operation. Include a fallback for empty lists (`if list else 0.0`) to avoid exceptions.
