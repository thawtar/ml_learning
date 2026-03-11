"""
Module 1 — Lesson 3: Linear Algebra with NumPy
================================================
Linear algebra is the language of ML. Matrix multiplication,
eigenvalues, SVD — these power everything from linear regression
to PCA to neural networks.
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════
# 1. DOT PRODUCT & MATRIX MULTIPLICATION
# ══════════════════════════════════════════════════════════════════════
print("── Dot product & matmul ──")

# Vector dot product: sum of element-wise products
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("dot(a, b):", np.dot(a, b))    # 1*4 + 2*5 + 3*6 = 32
print("a @ b:", a @ b)                # @ operator (preferred)

# Matrix multiplication
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
print("\nA @ B:\n", A @ B)            # (2,2) @ (2,2) → (2,2)

# Matrix-vector multiply
x = np.array([1, 2])
print("A @ x:", A @ x)               # (2,2) @ (2,) → (2,)

# ⚠ IMPORTANT: * is element-wise, @ is matrix multiplication
print("\nA * B (element-wise):\n", A * B)
print("A @ B (matrix multiply):\n", A @ B)

# Batch matrix multiply for ML
# X: (n_samples, n_features), W: (n_features, n_outputs)
X = np.random.randn(100, 5)     # 100 samples, 5 features
W = np.random.randn(5, 3)       # weight matrix
bias = np.array([0.1, 0.2, 0.3])
Y = X @ W + bias                # (100,5) @ (5,3) + (3,) = (100,3)
print(f"\nDense layer: X{X.shape} @ W{W.shape} + b{bias.shape} → Y{Y.shape}")


# ══════════════════════════════════════════════════════════════════════
# 2. MATRIX PROPERTIES & OPERATIONS
# ══════════════════════════════════════════════════════════════════════
print("\n── Matrix operations ──")

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])  # using 10 to make it invertible

print("Transpose A.T:\n", A.T)
print("Trace (sum of diagonal):", np.trace(A))
print("Determinant:", np.linalg.det(A))
print("Rank:", np.linalg.matrix_rank(A))

# Inverse
A_inv = np.linalg.inv(A)
print("\nInverse:\n", A_inv.round(4))
print("A @ A_inv ≈ I:\n", (A @ A_inv).round(10))

# Solve linear system: Ax = b  →  x = A⁻¹b
b = np.array([1, 2, 3])
x = np.linalg.solve(A, b)   # more numerically stable than inv(A) @ b
print("\nSolving Ax = b:")
print(f"  x = {x.round(4)}")
print(f"  Verify A @ x = {(A @ x).round(10)}")


# ══════════════════════════════════════════════════════════════════════
# 3. EIGENVALUES & EIGENVECTORS
# ══════════════════════════════════════════════════════════════════════
print("\n── Eigendecomposition ──")

# Symmetric matrix (common in ML: covariance matrices)
C = np.array([[4, 2],
              [2, 3]])

eigenvalues, eigenvectors = np.linalg.eigh(C)   # eigh for symmetric
print("Eigenvalues:", eigenvalues)
print("Eigenvectors (columns):\n", eigenvectors)

# Verify: C @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"  C @ v{i} = {(C @ v).round(6)}, λ*v{i} = {(lam * v).round(6)}")

# ML Application: PCA from scratch
print("\n── PCA from scratch ──")
np.random.seed(42)
X = np.random.randn(200, 3)
# Make features correlated
X[:, 1] = X[:, 0] * 0.8 + X[:, 1] * 0.2
X[:, 2] = X[:, 0] * 0.5 + X[:, 2] * 0.5

# 1. Center the data
X_centered = X - X.mean(axis=0)

# 2. Compute covariance matrix
cov = (X_centered.T @ X_centered) / (len(X) - 1)  # or np.cov(X.T)
print("Covariance matrix:\n", cov.round(4))

# 3. Eigendecompose
eigenvalues, eigenvectors = np.linalg.eigh(cov)
# Sort by descending eigenvalue
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Explained variance:", eigenvalues.round(4))
print("Explained variance ratio:", (eigenvalues / eigenvalues.sum()).round(4))

# 4. Project to 2D
X_pca = X_centered @ eigenvectors[:, :2]
print(f"Projected shape: {X_pca.shape}")  # (200, 2)


# ══════════════════════════════════════════════════════════════════════
# 4. SINGULAR VALUE DECOMPOSITION (SVD)
# ══════════════════════════════════════════════════════════════════════
print("\n── SVD ──")

# A = U @ diag(S) @ Vt
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")
print("Singular values:", S.round(4))

# Reconstruct: A ≈ U @ diag(S) @ Vt
A_reconstructed = U @ np.diag(S) @ Vt
print("Reconstruction error:", np.linalg.norm(A - A_reconstructed))

# Low-rank approximation (keep top k singular values)
k = 2
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"Rank-{k} approximation error: {np.linalg.norm(A - A_approx):.4f}")

# ML Application: SVD is used for:
# - Dimensionality reduction (truncated SVD / PCA)
# - Recommendation systems (matrix factorization)
# - Image compression
# - Computing pseudoinverse (lstsq)


# ══════════════════════════════════════════════════════════════════════
# 5. NORMS & DISTANCES
# ══════════════════════════════════════════════════════════════════════
print("\n── Norms ──")

v = np.array([3.0, 4.0])
print("L2 norm (Euclidean):", np.linalg.norm(v))           # sqrt(9+16) = 5
print("L1 norm (Manhattan):", np.linalg.norm(v, ord=1))    # 3+4 = 7
print("L∞ norm (max):", np.linalg.norm(v, ord=np.inf))     # 4

# Matrix norms
A = np.array([[1, 2], [3, 4]])
print("\nFrobenius norm:", np.linalg.norm(A, 'fro'))  # sqrt(sum of squares)

# Cosine similarity (common in NLP / embedding spaces)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 4, 6])
vec3 = np.array([-1, -2, -3])
print(f"\ncos_sim(v1, v2): {cosine_similarity(vec1, vec2):.4f}")   # 1.0 (same direction)
print(f"cos_sim(v1, v3): {cosine_similarity(vec1, vec3):.4f}")     # -1.0 (opposite)


# ══════════════════════════════════════════════════════════════════════
# 6. LEAST SQUARES (Linear Regression from scratch)
# ══════════════════════════════════════════════════════════════════════
print("\n── Least squares regression ──")
np.random.seed(42)

# Generate data: y = 3*x + 7 + noise
x = np.linspace(0, 10, 50)
y = 3 * x + 7 + np.random.randn(50) * 2

# Build design matrix: [x, 1]
X = np.column_stack([x, np.ones_like(x)])

# Normal equation: w = (X.T @ X)⁻¹ @ X.T @ y
w = np.linalg.solve(X.T @ X, X.T @ y)
print(f"Estimated: y = {w[0]:.2f}*x + {w[1]:.2f}")  # should be ~3, ~7

# Or use lstsq (handles rank-deficient matrices):
w2, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
print(f"lstsq:     y = {w2[0]:.2f}*x + {w2[1]:.2f}")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 3.1: Implement a function that computes the covariance matrix
    of a dataset X (n_samples, n_features) from scratch using matrix
    multiplication. Compare with np.cov(X.T).

Exercise 3.2: Perform full PCA on the Iris dataset (sklearn.datasets.load_iris):
    - Center the data
    - Compute eigenvectors of the covariance matrix
    - Project onto the first 2 principal components
    - Print the explained variance ratio for each component

Exercise 3.3: Given a 100×50 matrix, use SVD to create rank-5 and rank-10
    approximations. Compute the reconstruction error for each. How much
    of the Frobenius norm is captured?

Exercise 3.4: Implement cosine similarity matrix for a set of N vectors
    of dimension D, returning an (N, N) matrix. Use matrix operations only.
    Hint: normalize each row, then compute X @ X.T

Exercise 3.5: Solve a polynomial regression problem using least squares:
    Fit y = ax² + bx + c by constructing the appropriate design matrix
    X = [x², x, 1] and solving the normal equation.
"""
