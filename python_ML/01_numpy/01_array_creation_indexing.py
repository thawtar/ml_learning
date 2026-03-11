"""
Module 1 — Lesson 1: NumPy Array Creation & Indexing
=====================================================
NumPy is the bedrock of the Python ML stack. Every library
(Pandas, Scikit-learn, PyTorch) uses NumPy arrays internally.

Key concepts:
  - ndarray: N-dimensional, homogeneous, fixed-size container
  - dtype: the data type of every element (float64, int32, etc.)
  - shape: tuple describing dimensions — e.g. (3, 4) = 3 rows, 4 cols
  - Indexing: basic, slicing, fancy (integer array), boolean masking
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════
# 1. ARRAY CREATION
# ══════════════════════════════════════════════════════════════════════

# From Python lists
a = np.array([1, 2, 3, 4])
print("1D array:", a, "| shape:", a.shape, "| dtype:", a.dtype)

b = np.array([[1, 2, 3],
              [4, 5, 6]])
print("2D array:\n", b, "| shape:", b.shape)

# Specifying dtype explicitly (common in ML for memory control)
c = np.array([1, 2, 3], dtype=np.float32)
print("float32 array:", c, "| dtype:", c.dtype)

# ── Common factory functions ─────────────────────────────────────────
print("\n── Factory functions ──")
print("zeros:  ", np.zeros((2, 3)))            # 2×3 of 0.0
print("ones:   ", np.ones((3,), dtype=int))    # 1D of 1s
print("full:   ", np.full((2, 2), 7.0))        # 2×2 of 7.0
print("eye:    \n", np.eye(3))                  # 3×3 identity
print("arange: ", np.arange(0, 10, 2))         # [0, 2, 4, 6, 8]
print("linspace:", np.linspace(0, 1, 5))       # 5 evenly spaced in [0,1]

# ── zeros_like / ones_like (match shape & dtype of existing array) ───
template = np.array([[1.0, 2.0], [3.0, 4.0]])
print("zeros_like:", np.zeros_like(template))

# ── Reshape ──────────────────────────────────────────────────────────
print("\n── Reshaping ──")
flat = np.arange(12)
print("flat:", flat)
grid = flat.reshape(3, 4)       # 3 rows × 4 cols
print("reshaped (3,4):\n", grid)
print("reshaped (-1,6):\n", flat.reshape(-1, 6))  # -1 = infer dimension

# Flatten back
print("flatten:", grid.flatten())   # always returns a copy
print("ravel:  ", grid.ravel())     # returns a view when possible


# ══════════════════════════════════════════════════════════════════════
# 2. BASIC INDEXING & SLICING
# ══════════════════════════════════════════════════════════════════════
print("\n── Basic indexing ──")
arr = np.arange(10)               # [0, 1, 2, ..., 9]
print("arr[3]:", arr[3])           # single element
print("arr[2:7]:", arr[2:7])       # slice [2,3,4,5,6]
print("arr[::2]:", arr[::2])       # every other [0,2,4,6,8]
print("arr[-3:]:", arr[-3:])       # last 3: [7,8,9]

# 2D indexing: [row, col]
mat = np.arange(12).reshape(3, 4)
print("\nMatrix:\n", mat)
print("mat[1, 2]:", mat[1, 2])         # row 1, col 2 → 6
print("mat[0, :]:", mat[0, :])         # entire row 0
print("mat[:, 1]:", mat[:, 1])         # entire col 1
print("mat[1:, :2]:\n", mat[1:, :2])  # rows 1+, cols 0-1

# ⚠ IMPORTANT: slices return VIEWS, not copies!
view = arr[3:6]
view[0] = 999
print("After modifying view, arr:", arr)   # arr[3] is now 999!
# Use .copy() to avoid this: safe = arr[3:6].copy()


# ══════════════════════════════════════════════════════════════════════
# 3. FANCY INDEXING (integer array indexing)
# ══════════════════════════════════════════════════════════════════════
print("\n── Fancy indexing ──")
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 3, 4])
print("arr[indices]:", arr[indices])   # [10, 40, 50]

# 2D fancy indexing — select specific (row, col) pairs
mat = np.arange(12).reshape(3, 4)
rows = np.array([0, 1, 2])
cols = np.array([1, 3, 0])
print("mat[rows, cols]:", mat[rows, cols])  # mat[0,1], mat[1,3], mat[2,0]

# Select entire rows by index
print("Select rows [0,2]:\n", mat[[0, 2]])

# ⚠ Fancy indexing always returns a COPY (unlike slicing)


# ══════════════════════════════════════════════════════════════════════
# 4. BOOLEAN MASKING
# ══════════════════════════════════════════════════════════════════════
print("\n── Boolean masking ──")
data = np.array([1, -2, 3, -4, 5, -6])

mask = data > 0
print("Mask (data > 0):", mask)           # [True, False, True, ...]
print("Positive values:", data[mask])     # [1, 3, 5]
print("Negative values:", data[~mask])    # [-2, -4, -6] (~ = NOT)

# Combined conditions (use & for AND, | for OR, ~ for NOT)
arr = np.arange(20)
result = arr[(arr > 5) & (arr < 15)]  # Parentheses are required!
print("5 < arr < 15:", result)

# np.where — conditional selection
labels = np.where(data > 0, "pos", "neg")
print("Labels:", labels)

# Replace negatives with 0 (common in ML: ReLU-style clipping)
clipped = np.where(data > 0, data, 0)
print("Clipped:", clipped)


# ══════════════════════════════════════════════════════════════════════
# 5. USEFUL SHAPE OPERATIONS FOR ML
# ══════════════════════════════════════════════════════════════════════
print("\n── Shape operations ──")

# Add a dimension (e.g., turn (N,) into (N,1) for sklearn)
vec = np.array([1, 2, 3])
print("Original shape:", vec.shape)              # (3,)
print("np.newaxis col:", vec[:, np.newaxis].shape)  # (3, 1)
print("np.newaxis row:", vec[np.newaxis, :].shape)  # (1, 3)
# Equivalent: vec.reshape(-1, 1)

# Transpose
mat = np.arange(6).reshape(2, 3)
print("Original (2,3):\n", mat)
print("Transposed (3,2):\n", mat.T)

# Stacking
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("vstack:\n", np.vstack([a, b]))   # (2,3) — stack as rows
print("hstack:", np.hstack([a, b]))      # (6,)  — concatenate
print("column_stack:\n", np.column_stack([a, b]))  # (3,2) — as columns


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 1.1: Create a 5×5 matrix with values 1-25. Extract the 3×3
    center submatrix using slicing.

Exercise 1.2: Given `scores = np.array([85, 42, 91, 67, 73, 55, 38, 96])`,
    use boolean masking to find all scores >= 70 (passing).
    Then use np.where to replace failing scores with "FAIL" and passing
    scores with "PASS".

Exercise 1.3: Create a (10,) array of random integers 0-9.
    Use fancy indexing to select elements at even indices.
    Then use boolean masking to select only the values > 5.

Exercise 1.4: Reshape a 1D array of 24 elements into shapes:
    (2,12), (3,8), (4,6), (2,3,4). Verify each with .shape.

Exercise 1.5 (ML context): Given feature vectors as rows:
    X = np.random.randn(100, 5)   # 100 samples, 5 features
    - Extract the first feature (column 0) as a (100,1) column vector
    - Select samples where feature 0 > 1.0 (outlier detection)
    - Count how many samples have ALL features positive
"""
