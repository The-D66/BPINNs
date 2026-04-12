
## 2024-05-18 - [Optimizing Theta class sum of squares]
**Learning:** Python's `sum()` combined with `tf.norm(t)**2` computes a square root only to square it right back which is highly inefficient for summing tensor sizes within TF.
**Action:** Use `tf.add_n([tf.reduce_sum(tf.square(t)) for t in self.values])` to compute the sum of squares, doing all the work in tensorflow's native backend without computing the redundant square root.
