## 2024-05-18 - [Python sum() vs tf.add_n() for Tensors]
**Learning:** Using Python's built-in `sum()` over an iterable of TensorFlow tensors generates an unnecessarily large computation graph and creates numerous intermediate tensor objects, slowing down graph execution significantly.
**Action:** When summing over an iterable of tensors (like a list of dictionary values containing loss terms), always use `tf.add_n()` for a single, optimized node in the computation graph. Always include a fallback like `tf.add_n(vals) if vals else tf.constant(0.0)` for edge cases involving empty iterables.
