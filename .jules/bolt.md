## 2025-05-09 - [Optimize tf tensor aggregations]
**Learning:** `sum` combined with `tf.norm(t)**2` for aggregating tf tensors creates high build overhead.
**Action:** Use native TensorFlow operators: `tf.add_n([tf.reduce_sum(tf.square(t)) for t in values])` for optimal performance without unnecessary roots.
