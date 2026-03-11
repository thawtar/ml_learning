"""
Module 1 — Lesson 4: Random Number Generation & Statistics
===========================================================
ML relies heavily on randomness: data splitting, weight initialization,
sampling, bootstrapping, and stochastic algorithms.
NumPy's random module is the standard tool.
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════
# 1. MODERN RNG (Generator API — recommended since NumPy 1.17)
# ══════════════════════════════════════════════════════════════════════
print("── Modern RNG ──")

# Create a reproducible random generator
rng = np.random.default_rng(seed=42)

# Basic distributions
print("Uniform [0,1):", rng.random(5).round(3))
print("Uniform [a,b):", rng.uniform(low=2, high=5, size=5).round(3))
print("Integers [0,10):", rng.integers(0, 10, size=5))
print("Normal(0,1):", rng.standard_normal(5).round(3))
print("Normal(μ=5,σ=2):", rng.normal(loc=5, scale=2, size=5).round(3))

# ⚠ Avoid legacy API: np.random.rand(), np.random.randn()
# The Generator API is faster, statistically better, and thread-safe.

# ── Reproducibility ──────────────────────────────────────────────────
rng1 = np.random.default_rng(seed=42)
rng2 = np.random.default_rng(seed=42)
print("\nSame seed, same results:", np.array_equal(rng1.random(5), rng2.random(5)))


# ══════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTIONS FOR ML
# ══════════════════════════════════════════════════════════════════════
print("\n── Distributions ──")
rng = np.random.default_rng(42)

# Uniform — random initialization, dropout masks
print("Uniform(0,1):", rng.uniform(0, 1, 5).round(3))

# Normal — weight initialization (Gaussian init)
print("Normal(0, 0.01):", rng.normal(0, 0.01, 5).round(4))

# Xavier/Glorot initialization: scale = sqrt(2 / (fan_in + fan_out))
fan_in, fan_out = 784, 256
scale = np.sqrt(2.0 / (fan_in + fan_out))
weights = rng.normal(0, scale, size=(fan_in, fan_out))
print(f"Xavier init: shape={weights.shape}, std={weights.std():.4f}, expected={scale:.4f}")

# He initialization: scale = sqrt(2 / fan_in) — for ReLU networks
scale_he = np.sqrt(2.0 / fan_in)
weights_he = rng.normal(0, scale_he, size=(fan_in, fan_out))
print(f"He init: std={weights_he.std():.4f}, expected={scale_he:.4f}")

# Binomial — simulating coin flips, dropout
dropout_mask = rng.binomial(1, p=0.8, size=10)  # keep 80% of neurons
print("Dropout mask:", dropout_mask)

# Choice — sampling from a set
classes = ["cat", "dog", "bird"]
samples = rng.choice(classes, size=5, replace=True)
print("Random samples:", samples)

# Weighted sampling
probs = [0.6, 0.3, 0.1]
samples = rng.choice(classes, size=10, p=probs)
print("Weighted samples:", samples)


# ══════════════════════════════════════════════════════════════════════
# 3. SHUFFLING & SAMPLING
# ══════════════════════════════════════════════════════════════════════
print("\n── Shuffling & sampling ──")
rng = np.random.default_rng(42)

# Shuffle in place
arr = np.arange(10)
rng.shuffle(arr)
print("Shuffled:", arr)

# Permutation (returns new array, original unchanged)
arr = np.arange(10)
perm = rng.permutation(arr)
print("Permuted:", perm)
print("Original:", arr)   # unchanged

# ── Train/test split (from scratch) ─────────────────────────────────
X = np.arange(100).reshape(20, 5)   # 20 samples, 5 features
y = np.arange(20)

indices = rng.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ── Bootstrapping ────────────────────────────────────────────────────
data = np.array([2.5, 3.1, 4.2, 3.7, 5.0, 2.8, 4.5, 3.9])
n_bootstrap = 1000
boot_means = np.array([
    rng.choice(data, size=len(data), replace=True).mean()
    for _ in range(n_bootstrap)
])
print(f"\nBootstrap mean: {boot_means.mean():.3f} ± {boot_means.std():.3f}")
print(f"95% CI: [{np.percentile(boot_means, 2.5):.3f}, {np.percentile(boot_means, 97.5):.3f}]")


# ══════════════════════════════════════════════════════════════════════
# 4. STATISTICAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════
print("\n── Statistics ──")
rng = np.random.default_rng(42)

data = rng.normal(loc=100, scale=15, size=1000)  # IQ-like distribution

print(f"Mean:   {data.mean():.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Std:    {data.std():.2f}")
print(f"Var:    {data.var():.2f}")
print(f"Min:    {data.min():.2f}")
print(f"Max:    {data.max():.2f}")

# Percentiles / quantiles
print(f"\n25th percentile: {np.percentile(data, 25):.2f}")
print(f"50th percentile: {np.percentile(data, 50):.2f}")
print(f"75th percentile: {np.percentile(data, 75):.2f}")
print(f"IQR: {np.percentile(data, 75) - np.percentile(data, 25):.2f}")

# Correlation
x = rng.standard_normal(100)
y = 0.8 * x + 0.2 * rng.standard_normal(100)  # correlated
corr_matrix = np.corrcoef(x, y)
print(f"\nCorrelation matrix:\n{corr_matrix.round(4)}")
print(f"Pearson r: {corr_matrix[0, 1]:.4f}")

# Histogram (counts)
counts, bin_edges = np.histogram(data, bins=10)
print(f"\nHistogram: {counts}")

# Unique values and counts (for classification labels)
labels = rng.integers(0, 5, size=100)
unique, counts = np.unique(labels, return_counts=True)
print(f"\nClass distribution: {dict(zip(unique, counts))}")


# ══════════════════════════════════════════════════════════════════════
# 5. SORTING & SEARCHING
# ══════════════════════════════════════════════════════════════════════
print("\n── Sorting & searching ──")

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("sorted:", np.sort(arr))
print("argsort:", np.argsort(arr))          # indices that would sort
print("top-3 indices:", np.argsort(arr)[-3:][::-1])  # top-3 largest

# Partition (faster than full sort when you need top-k)
# np.partition puts the k smallest in the first k positions (unordered)
print("partition(k=3):", np.partition(arr, 3))

# searchsorted — binary search in sorted array
sorted_arr = np.sort(arr)
idx = np.searchsorted(sorted_arr, 4)
print(f"Insert 4 at index {idx} in {sorted_arr}")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 4.1: Generate 10,000 samples from a normal distribution
    with mean=0, std=1. Verify empirically that:
    - ~68% fall within [-1, 1]
    - ~95% fall within [-2, 2]
    - ~99.7% fall within [-3, 3]

Exercise 4.2: Implement a function `stratified_split(X, y, test_ratio=0.2)`
    that splits data while preserving class proportions. Use rng.permutation
    within each class. Verify class ratios match between train and test.

Exercise 4.3: Implement k-fold cross validation indices from scratch.
    Given n=100 samples and k=5 folds, return a list of (train_idx, val_idx)
    tuples.

Exercise 4.4: Generate data from two Gaussian clusters:
    - Cluster A: mean=[0,0], std=1, n=100
    - Cluster B: mean=[5,5], std=1, n=100
    Compute the mean and covariance of each cluster.
    Compute the distance between cluster centers.

Exercise 4.5: Implement a simple Monte Carlo estimation of π:
    Generate random points in [0,1]×[0,1], count how many fall
    inside the quarter circle (x²+y²≤1), estimate π = 4 * (inside/total).
"""
