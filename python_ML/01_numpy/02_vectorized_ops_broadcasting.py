"""
Module 1 — Lesson 2: Vectorized Operations & Broadcasting
==========================================================
The #1 rule of NumPy: NEVER loop over elements in Python.
Vectorized operations run in compiled C, often 10-100× faster.

Broadcasting is NumPy's mechanism for applying operations between
arrays of different shapes — understanding it is essential for ML.
"""

import numpy as np
import time

# ══════════════════════════════════════════════════════════════════════
# 1. VECTORIZED (ELEMENT-WISE) OPERATIONS
# ══════════════════════════════════════════════════════════════════════
print("── Vectorized ops ──")

a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([10.0, 20.0, 30.0, 40.0])

print("a + b:", a + b)       # element-wise addition
print("a * b:", a * b)       # element-wise multiplication (NOT dot product)
print("a ** 2:", a ** 2)     # element-wise square
print("a / b:", a / b)       # element-wise division
print("np.sqrt(a):", np.sqrt(a))
print("np.exp(a):", np.exp(a))
print("np.log(a):", np.log(a))

# Comparison operators return boolean arrays
print("a > 2:", a > 2)       # [False, False,  True,  True]

# ── Universal Functions (ufuncs) ─────────────────────────────────────
# NumPy ufuncs are the building blocks. Common ones for ML:
x = np.linspace(-3, 3, 7)
print("\nx:", x)
print("np.abs(x):", np.abs(x))
print("np.maximum(x, 0):", np.maximum(x, 0))    # ReLU!
print("np.clip(x, -1, 1):", np.clip(x, -1, 1))  # Clamp to range

# Sigmoid function (vectorized)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("sigmoid(x):", sigmoid(x))

# Softmax (vectorized, numerically stable)
def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum()

logits = np.array([2.0, 1.0, 0.1])
print("softmax:", softmax(logits))
print("sum:", softmax(logits).sum())  # should be 1.0


# ══════════════════════════════════════════════════════════════════════
# 2. WHY VECTORIZATION MATTERS — SPEED COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n── Speed comparison ──")

n = 1_000_000
a = np.random.randn(n)
b = np.random.randn(n)

# Python loop
start = time.perf_counter()
result_loop = [a[i] + b[i] for i in range(n)]
loop_time = time.perf_counter() - start

# NumPy vectorized
start = time.perf_counter()
result_vec = a + b
vec_time = time.perf_counter() - start

print(f"Python loop: {loop_time:.4f}s")
print(f"NumPy vectorized: {vec_time:.6f}s")
print(f"Speedup: {loop_time / vec_time:.0f}×")


# ══════════════════════════════════════════════════════════════════════
# 3. AGGREGATIONS / REDUCTIONS
# ══════════════════════════════════════════════════════════════════════
print("\n── Aggregations ──")

mat = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("sum (all):", mat.sum())           # 45
print("sum (rows, axis=0):", mat.sum(axis=0))   # [12, 15, 18] — collapse rows
print("sum (cols, axis=1):", mat.sum(axis=1))   # [6, 15, 24]  — collapse cols

# axis=0 → operate DOWN columns (collapse rows)
# axis=1 → operate ACROSS columns (collapse cols)
# Think: axis=N means "eliminate dimension N"

print("mean (cols):", mat.mean(axis=1))
print("std (all):", mat.std())
print("max (rows):", mat.max(axis=0))
print("argmax (all):", mat.argmax())       # index of max element (flat)
print("argmax (rows):", mat.argmax(axis=0)) # index of max in each column

# ── keepdims — preserve dimensions for broadcasting ──────────────────
row_means = mat.mean(axis=1, keepdims=True)  # shape (3,1) not (3,)
print("row means (keepdims):", row_means.shape)
centered = mat - row_means  # broadcasting works because (3,3) - (3,1)
print("Centered matrix:\n", centered)


# ══════════════════════════════════════════════════════════════════════
# 4. BROADCASTING
# ══════════════════════════════════════════════════════════════════════
print("\n── Broadcasting ──")

# Broadcasting rules:
# 1. If arrays differ in ndim, prepend 1s to the smaller shape
# 2. Dimensions of size 1 are stretched to match the other
# 3. If dimensions differ and neither is 1 → error

# Example 1: scalar + array
a = np.array([1, 2, 3])
print("a + 10:", a + 10)     # 10 is broadcast to [10, 10, 10]

# Example 2: (3,4) + (4,) → each row gets the vector added
mat = np.ones((3, 4))
row = np.array([1, 2, 3, 4])
print("(3,4) + (4,):\n", mat + row)

# Example 3: (3,1) + (1,4) → outer operation → (3,4)
col = np.array([[10], [20], [30]])    # shape (3,1)
row = np.array([1, 2, 3, 4])          # shape (4,) → broadcast to (1,4)
print("(3,1) + (1,4):\n", col + row)   # shape (3,4)

# ── ML Application: Feature Normalization ────────────────────────────
print("\n── ML: Feature normalization ──")
# X has shape (n_samples, n_features)
X = np.random.randn(5, 3) * 10 + 50

print("Before normalization:")
print(f"  means: {X.mean(axis=0)}")
print(f"  stds:  {X.std(axis=0)}")

# Standardize: (X - mean) / std   (broadcast: (5,3) - (3,) / (3,))
means = X.mean(axis=0)     # shape (3,)
stds = X.std(axis=0)       # shape (3,)
X_normalized = (X - means) / stds

print("After normalization:")
print(f"  means: {X_normalized.mean(axis=0).round(10)}")  # ~0
print(f"  stds:  {X_normalized.std(axis=0).round(10)}")   # ~1


# ── Broadcasting Shape Visualization ─────────────────────────────────
# Watch shapes:
#   (5, 3)  X
# - (   3,) means    → means broadcast to (1,3) then (5,3) ✓
# / (   3,) stds     → same
# = (5, 3)  result

# Another example:
# (256, 256, 3) image     (H, W, channels)
# *         (3,) weights  → broadcast to (1, 1, 3) → (256, 256, 3)
# Common for channel-wise image operations!


# ══════════════════════════════════════════════════════════════════════
# 5. COMMON PATTERNS IN ML
# ══════════════════════════════════════════════════════════════════════
print("\n── ML patterns ──")

# Euclidean distance between two vectors
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
dist = np.sqrt(np.sum((a - b) ** 2))
print(f"Euclidean distance: {dist:.4f}")
# Or: np.linalg.norm(a - b)

# Pairwise distances (using broadcasting) — shape (n, m)
# Given points X (n,d) and Y (m,d):
X = np.random.randn(3, 2)  # 3 points in 2D
Y = np.random.randn(4, 2)  # 4 points in 2D
# Expand dims: X → (3,1,2), Y → (1,4,2) → diff is (3,4,2)
diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
dists = np.sqrt((diff ** 2).sum(axis=2))  # (3, 4)
print("Pairwise distance matrix shape:", dists.shape)

# One-hot encoding
labels = np.array([0, 2, 1, 0, 3])
n_classes = 4
one_hot = np.eye(n_classes)[labels]
print("One-hot:\n", one_hot)


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 2.1: Implement a ReLU function: relu(x) = max(0, x) using
    np.maximum. Test on np.array([-3, -1, 0, 1, 3]).

Exercise 2.2: Given a matrix X of shape (100, 5), normalize each
    column to the range [0, 1] using: (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    Verify min is 0 and max is 1 for each column.

Exercise 2.3: Given two sets of points A (10, 3) and B (20, 3),
    compute the full pairwise Euclidean distance matrix of shape (10, 20)
    using broadcasting (no loops!).

Exercise 2.4: Implement batch matrix-vector multiply:
    Given W of shape (d_out, d_in) and X of shape (batch, d_in),
    compute Y = X @ W.T  so Y has shape (batch, d_out).
    This is a dense layer forward pass!

Exercise 2.5: Write a vectorized MSE loss function:
    mse(y_true, y_pred) = mean((y_true - y_pred)²)
    Test it on random arrays of shape (100,).
"""
