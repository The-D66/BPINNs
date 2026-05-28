
## 2024-05-28 - [Performance: Python math.prod vs np.prod for small structures]
**Learning:** For small arrays or tuples (like tensor shapes), Python's `math.prod` is significantly faster than `np.prod` as it avoids NumPy overhead and type conversion.
**Action:** Default to `math.prod` over `np.prod` when dealing with simple dimension calculations and generating configuration data.
