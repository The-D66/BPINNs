## 2024-06-25 - Python's `math.prod` vs `np.prod` for list of integers
**Learning:** Using `math.prod` instead of `np.prod` to calculate the product of elements in small, simple tuples/lists (like tensor shapes) avoids unnecessary NumPy type conversion overhead and is significantly faster.
**Action:** Prefer Python's built-in `math.prod` when calculating the product of elements in small, simple tuples/lists.
