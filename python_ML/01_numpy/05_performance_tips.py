"""
Module 1 — Lesson 5: NumPy Performance Tips
=============================================
Writing fast NumPy code is critical for ML — the difference between
a training loop taking seconds vs. hours. This lesson covers memory
layout, views vs copies, and vectorization strategies.
"""

import numpy as np
import time

def benchmark(label, func, *args, n_runs=100):
    """Quick benchmark utility."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    avg = np.mean(times) * 1000
    print(f"  {label}: {avg:.3f} ms (avg of {n_runs} runs)")
    return avg


# ══════════════════════════════════════════════════════════════════════
# 1. MEMORY LAYOUT: C-ORDER vs FORTRAN-ORDER
# ══════════════════════════════════════════════════════════════════════
print("── Memory layout ──")

# C-order (row-major): rows are contiguous in memory — NumPy default
# Fortran-order (column-major): columns are contiguous

a_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')
a_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')

print("C-order contiguous:", a_c.flags['C_CONTIGUOUS'])    # True
print("F-order contiguous:", a_f.flags['F_CONTIGUOUS'])    # True

# Why it matters: iterating along the contiguous axis is much faster
# (better CPU cache utilization)
big = np.random.randn(1000, 1000)

# Row-wise sum (fast in C-order — follows memory layout)
t1 = benchmark("Row-wise sum (axis=1)", lambda: big.sum(axis=1), n_runs=200)
# Col-wise sum (crosses rows in memory — slower in C-order)
t2 = benchmark("Col-wise sum (axis=0)", lambda: big.sum(axis=0), n_runs=200)


# ══════════════════════════════════════════════════════════════════════
# 2. VIEWS vs COPIES
# ══════════════════════════════════════════════════════════════════════
print("\n── Views vs copies ──")

arr = np.arange(12).reshape(3, 4)

# VIEWS (no data copy — share memory with original)
view1 = arr[1:3]           # slicing → view
view2 = arr.reshape(4, 3)  # reshape → view (usually)
view3 = arr.T              # transpose → view

print("Slice is view:", view1.base is arr)
print("Reshape is view:", view2.base is arr)

# COPIES (new memory allocation)
copy1 = arr[[0, 2]]         # fancy indexing → copy
copy2 = arr.copy()           # explicit copy
copy3 = arr.flatten()        # flatten → copy (ravel → view when possible)

print("Fancy index is copy:", copy1.base is None)

# ⚠ Modifying a view modifies the original!
view1[0, 0] = 999
print("After modifying view, arr[1,0]:", arr[1, 0])  # 999!

# Tip: use .copy() when you need independent data
safe = arr[1:3].copy()
safe[0, 0] = 0  # doesn't affect arr


# ══════════════════════════════════════════════════════════════════════
# 3. AVOIDING PYTHON LOOPS
# ══════════════════════════════════════════════════════════════════════
print("\n── Loop elimination ──")

n = 100_000
a = np.random.randn(n)
b = np.random.randn(n)

# BAD: Python loop
def dot_loop(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

# GOOD: Vectorized
def dot_vec(a, b):
    return np.sum(a * b)

# BEST: Built-in
def dot_builtin(a, b):
    return np.dot(a, b)

benchmark("Python loop", dot_loop, a, b, n_runs=5)
benchmark("np.sum(a*b)", dot_vec, a, b, n_runs=100)
benchmark("np.dot(a,b)", dot_builtin, a, b, n_runs=100)


# ══════════════════════════════════════════════════════════════════════
# 4. VECTORIZATION STRATEGIES
# ══════════════════════════════════════════════════════════════════════
print("\n── Vectorization strategies ──")

# Strategy 1: Replace if/else with np.where
data = np.random.randn(10)
# BAD:  [x if x > 0 else 0 for x in data]
# GOOD: np.where(data > 0, data, 0)
print("ReLU:", np.where(data > 0, data, 0).round(3))

# Strategy 2: Replace accumulation loops with cumsum/cumprod
arr = np.array([1, 2, 3, 4, 5])
print("Cumulative sum:", np.cumsum(arr))      # [1, 3, 6, 10, 15]
print("Cumulative prod:", np.cumprod(arr))    # [1, 2, 6, 24, 120]

# Strategy 3: Use broadcasting instead of outer loops
# BAD:  [[a[i]*b[j] for j in range(m)] for i in range(n)]
# GOOD: a[:, np.newaxis] * b[np.newaxis, :]
a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])
outer = a[:, np.newaxis] * b[np.newaxis, :]
print("Outer product:\n", outer)

# Strategy 4: Use einsum for complex tensor operations
# Einstein summation — very flexible and often fast
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# Matrix multiply
C = np.einsum('ij,jk->ik', A, B)
print(f"\neinsum matmul: {C.shape}")

# Batch matrix multiply: (batch, m, n) @ (batch, n, p) → (batch, m, p)
batch_A = np.random.randn(10, 3, 4)
batch_B = np.random.randn(10, 4, 5)
batch_C = np.einsum('bij,bjk->bik', batch_A, batch_B)
print(f"einsum batch matmul: {batch_C.shape}")

# Trace
print(f"einsum trace: {np.einsum('ii->', np.eye(5))}")

# Diagonal
mat = np.arange(9).reshape(3, 3)
print(f"einsum diagonal: {np.einsum('ii->i', mat)}")


# ══════════════════════════════════════════════════════════════════════
# 5. MEMORY-EFFICIENT OPERATIONS
# ══════════════════════════════════════════════════════════════════════
print("\n── Memory efficiency ──")

# In-place operations save memory
a = np.random.randn(1_000_000)

# BAD (creates temporary):  a = a + 1
# GOOD (in-place):
a += 1
np.add(a, 1, out=a)   # explicit out parameter

# Pre-allocate output arrays
out = np.empty_like(a)
np.multiply(a, 2, out=out)

# Use appropriate dtypes
data_f64 = np.random.randn(1_000_000)                    # 8 bytes each
data_f32 = data_f64.astype(np.float32)                    # 4 bytes each
print(f"float64: {data_f64.nbytes / 1024:.0f} KB")
print(f"float32: {data_f32.nbytes / 1024:.0f} KB")

# For ML: float32 is usually sufficient and 2× less memory
# For deep learning: float16 (half precision) saves even more


# ══════════════════════════════════════════════════════════════════════
# 6. STRUCTURED ARRAYS (bonus — useful for tabular data w/o Pandas)
# ══════════════════════════════════════════════════════════════════════
print("\n── Structured arrays ──")

dt = np.dtype([('name', 'U20'), ('age', 'i4'), ('score', 'f8')])
people = np.array([
    ('Alice', 30, 92.5),
    ('Bob', 25, 87.3),
    ('Charlie', 35, 95.1),
], dtype=dt)

print("Names:", people['name'])
print("Scores:", people['score'])
print("Above 90:", people[people['score'] > 90])


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 5.1: Write a benchmark comparing:
    - Python list comprehension: [x**2 for x in range(1_000_000)]
    - np.arange(1_000_000) ** 2
    - np.square(np.arange(1_000_000))
    Which is fastest? Why?

Exercise 5.2: Given a large matrix X (10000, 100), compare the speed of:
    - X.sum(axis=0) vs X.sum(axis=1)
    - X.T @ X  vs  X @ X.T
    Explain the results in terms of memory layout.

Exercise 5.3: Rewrite this loop using vectorized NumPy:
    result = []
    for i in range(len(X)):
        if X[i] > threshold:
            result.append(X[i] * scale)
        else:
            result.append(X[i])

Exercise 5.4: Use np.einsum to compute:
    a) The trace of a matrix
    b) The diagonal of a matrix product A @ B
    c) The batch dot product of two (batch, dim) arrays
    Compare speed with explicit NumPy operations.

Exercise 5.5: Profile the memory usage of creating 1M objects:
    - As a list of Python tuples: [(x, y) for x, y in ...]
    - As a NumPy structured array
    - As a regular NumPy array of shape (1M, 2)
    Use sys.getsizeof and arr.nbytes to compare.
"""
