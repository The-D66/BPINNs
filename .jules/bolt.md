
## 2024-05-18 - Avoid numpy operations on simple Python tuples in performance critical code
**Learning:** Using `np.prod` to calculate the product of elements in a tuple (like a tensor's `shape`) introduces unnecessary type conversion overhead from Python tuple to NumPy array. For simple sequences where performance matters, Python's built-in `math.prod` is significantly faster. Similarly, `sum()` over TensorFlow tensors with overloaded operations creates Python loop overhead; offloading to TensorFlow via `tf.add_n` is much faster.
**Action:** Use `math.prod` instead of `np.prod` when computing the product of a small Python tuple (e.g., shape dimensions). Prefer `tf.add_n` over Python `sum` when combining many tensors in TensorFlow.
