## 2024-05-19 - NumPy prod overhead
**Learning:** `np.prod` has significant overhead compared to `math.prod` for small integer lists, likely due to numpy array instantiation and type conversions.
**Action:** Always prefer `math.prod` for calculating the product of elements in small, simple tuples/lists (like tensor shapes or domains).

## 2024-05-19 - Native TF summation
**Learning:** Combining native TF operators via `tf.add_n([tf.reduce_sum(tf.square(t)) for t in tensors])` is significantly faster (~3x) than python's `sum([tf.norm(t)**2 for t in tensors])`.
**Action:** Prefer `tf.add_n` when accumulating values across lists of tensors to avoid intermediate python objects. Always add a fallback (like `tf.constant(0.0)`) in case the list is empty to prevent crashes.

## 2024-05-19 - Generator expressions for sum
**Learning:** Using a generator expression with `sum(math.prod(t.shape) for t in self.values)` avoids unnecessary list allocation compared to `sum([math.prod(...) for ...])`.
**Action:** Use generator expressions when accumulating mapped values unless the intermediate list is needed elsewhere.
