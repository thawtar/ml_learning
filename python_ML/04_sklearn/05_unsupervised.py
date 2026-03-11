"""
Module 4 — Lesson 5: Unsupervised Learning
============================================
Clustering, dimensionality reduction, and anomaly detection.
These are essential for EDA, feature engineering, and data understanding.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ── Clustering ──
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# ── Evaluation ──
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    calinski_harabasz_score, davies_bouldin_score,
)

# ── Dimensionality Reduction ──
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE


# ══════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════
# Clean spherical clusters
X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, random_state=42)

# Non-spherical (moon shapes — harder for KMeans)
X_moons, y_moons = make_moons(n_samples=300, noise=0.08, random_state=42)

# Real data
iris = load_iris()
X_iris, y_iris = iris.data, iris.target


# ══════════════════════════════════════════════════════════════════════
# 1. K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════
print("── KMeans ──")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_blobs)

# Fit KMeans
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

print(f"Cluster sizes:        {np.bincount(labels)}")
print(f"Inertia (within-SS):  {kmeans.inertia_:.2f}")
print(f"Silhouette score:     {silhouette_score(X_scaled, labels):.4f}")
print(f"Adjusted Rand Index:  {adjusted_rand_score(y_blobs, labels):.4f}")


# ══════════════════════════════════════════════════════════════════════
# 2. ELBOW METHOD & SILHOUETTE — Choosing k
# ══════════════════════════════════════════════════════════════════════
print("\n── Choosing k ──")

print(f"{'k':>3s}  {'Inertia':>10s}  {'Silhouette':>11s}  {'Calinski-H':>11s}  {'Davies-B':>9s}")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labs = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labs)
    ch  = calinski_harabasz_score(X_scaled, labs)
    db  = davies_bouldin_score(X_scaled, labs)
    print(f"  {k}   {km.inertia_:>10.1f}   {sil:>10.4f}   {ch:>10.1f}   {db:>8.4f}")

"""
HOW TO CHOOSE k:

  Elbow method:  Plot inertia vs k → pick the "elbow" where it bends
  Silhouette:    Higher is better (max = 1). Measures compactness + separation.
  Calinski-Harabasz: Higher is better. Ratio of between/within variance.
  Davies-Bouldin:    Lower is better. Measures average cluster similarity.
  
  ⚠ These are heuristics. Domain knowledge often matters more.
"""


# ══════════════════════════════════════════════════════════════════════
# 3. DBSCAN — Density-Based Clustering
# ══════════════════════════════════════════════════════════════════════
print("\n── DBSCAN ──")

# DBSCAN shines on non-spherical clusters where KMeans fails
X_moon_sc = StandardScaler().fit_transform(X_moons)

# KMeans on moons: bad
km_moons = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_moon_sc)
print(f"KMeans ARI on moons:  {adjusted_rand_score(y_moons, km_moons.labels_):.4f}")

# DBSCAN on moons: good
db = DBSCAN(eps=0.3, min_samples=5)
db_labels = db.fit_predict(X_moon_sc)
n_noise = np.sum(db_labels == -1)
print(f"DBSCAN ARI on moons: {adjusted_rand_score(y_moons, db_labels):.4f}")
print(f"DBSCAN clusters found: {len(set(db_labels) - {-1})}, noise points: {n_noise}")

"""
DBSCAN PARAMETERS:
  eps:          Neighborhood radius. ↑ → larger clusters, fewer noise points.
  min_samples:  Minimum points to form a dense region. ↑ → more conservative.

  Pros:
    - No need to specify k
    - Finds arbitrary-shaped clusters
    - Identifies noise/outliers (label = -1)
  
  Cons:
    - Sensitive to eps and min_samples
    - Struggles with clusters of varying density
    
  Tip: Use k-distance graph to choose eps
    distances = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0]
    Sort and plot the distances to the k-th neighbor → look for the elbow.
"""


# ══════════════════════════════════════════════════════════════════════
# 4. GAUSSIAN MIXTURE MODELS
# ══════════════════════════════════════════════════════════════════════
print("\n── Gaussian Mixture ──")

gmm = GaussianMixture(n_components=4, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

print(f"ARI:          {adjusted_rand_score(y_blobs, gmm_labels):.4f}")
print(f"AIC:          {gmm.aic(X_scaled):.1f}")
print(f"BIC:          {gmm.bic(X_scaled):.1f}")

# Soft assignment: probability of belonging to each cluster
probs = gmm.predict_proba(X_scaled)
print(f"Sample 0 belongs to clusters with prob: {probs[0].round(3)}")

# Choose n_components by BIC
print("\n  Components  BIC")
for n in range(2, 8):
    g = GaussianMixture(n_components=n, random_state=42).fit(X_scaled)
    print(f"       {n}      {g.bic(X_scaled):.1f}")

"""
GMM vs KMeans:
  - GMM gives SOFT assignments (probabilities) — useful for uncertainty!
  - GMM can model elliptical clusters (different covariances)
  - KMeans assumes spherical, equal-size clusters
  - Use BIC or AIC to choose number of components
"""


# ══════════════════════════════════════════════════════════════════════
# 5. PCA — Principal Component Analysis
# ══════════════════════════════════════════════════════════════════════
print("\n── PCA ──")

X_iris_sc = StandardScaler().fit_transform(X_iris)

# Full PCA to see explained variance
pca_full = PCA().fit(X_iris_sc)
print("Explained variance ratio per component:")
for i, var in enumerate(pca_full.explained_variance_ratio_):
    cumulative = pca_full.explained_variance_ratio_[:i+1].sum()
    bar = "█" * int(var * 50)
    print(f"  PC{i+1}: {var:.4f}  (cumulative: {cumulative:.4f})  {bar}")

# Keep 95% variance
pca_95 = PCA(n_components=0.95)  # float → auto select n_components
X_pca = pca_95.fit_transform(X_iris_sc)
print(f"\nComponents needed for 95% variance: {pca_95.n_components_}")
print(f"Original shape: {X_iris_sc.shape} → PCA shape: {X_pca.shape}")

# 2D projection for visualization
pca_2d = PCA(n_components=2).fit_transform(X_iris_sc)
print(f"\n2D PCA - class separation (first 5 samples per class):")
for cls in range(3):
    mask = y_iris == cls
    centroid = pca_2d[mask].mean(axis=0)
    print(f"  Class {cls} ({iris.target_names[cls]:>10s}): centroid = ({centroid[0]:+.2f}, {centroid[1]:+.2f})")


# ══════════════════════════════════════════════════════════════════════
# 6. t-SNE — Non-Linear Dimensionality Reduction
# ══════════════════════════════════════════════════════════════════════
print("\n── t-SNE ──")

tsne = TSNE(
    n_components=2,
    perplexity=30,     # balances local vs global structure (5-50)
    random_state=42,
    n_iter=1000,
)
X_tsne = tsne.fit_transform(X_iris_sc)

print(f"t-SNE shape: {X_tsne.shape}")
for cls in range(3):
    mask = y_iris == cls
    centroid = X_tsne[mask].mean(axis=0)
    print(f"  Class {cls} ({iris.target_names[cls]:>10s}): "
          f"centroid = ({centroid[0]:+.2f}, {centroid[1]:+.2f})")

"""
PCA vs t-SNE:

Feature       │ PCA                       │ t-SNE
──────────────┼───────────────────────────┼──────────────────────────
Type          │ Linear                    │ Non-linear
Speed         │ Fast (O(n·d²))            │ Slow (O(n²))
Deterministic │ Yes                       │ No (random init)
New data      │ pca.transform(X_new)      │ ✗ Must refit (no transform)
Preserve      │ Global structure (variance)│ Local structure (neighbors)
Use for       │ Feature reduction, input  │ Visualization only!
              │ to models, denoising      │ ⚠ Never use for downstream ML

Tips:
  - Always scale data before PCA/t-SNE
  - For t-SNE, try perplexity = [5, 30, 50] and compare
  - For large datasets, use PCA first to reduce to ~50 dims, then t-SNE
  - UMAP is a faster alternative to t-SNE (pip install umap-learn)
"""


# ══════════════════════════════════════════════════════════════════════
# 7. PRACTICAL: Clustering + Dim. Reduction Pipeline
# ══════════════════════════════════════════════════════════════════════
print("\n── Practical: Reduce → Cluster → Evaluate ──")

# Step 1: PCA to reduce dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_iris_sc)

# Step 2: Cluster on reduced data
for name, algo in [
    ("KMeans",    KMeans(n_clusters=3, n_init=10, random_state=42)),
    ("DBSCAN",    DBSCAN(eps=0.7, min_samples=5)),
    ("GMM",       GaussianMixture(n_components=3, random_state=42)),
    ("Agglom.",   AgglomerativeClustering(n_clusters=3)),
]:
    labels = algo.fit_predict(X_reduced)
    n_clusters = len(set(labels) - {-1})
    if n_clusters >= 2:
        sil = silhouette_score(X_reduced, labels)
        ari = adjusted_rand_score(y_iris, labels)
        print(f"  {name:>10s}: clusters={n_clusters}  Silhouette={sil:.3f}  ARI={ari:.3f}")
    else:
        print(f"  {name:>10s}: only {n_clusters} cluster found (adjust params)")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 5.1: Generate data with make_blobs (5 clusters). Apply the
    elbow method + silhouette score to determine k. Does it find 5?

Exercise 5.2: Generate make_moons data with noise=0.15. Compare
    KMeans, DBSCAN, and Agglomerative clustering. Which works best?
    Report ARI for each.

Exercise 5.3: Load the Iris dataset. Use PCA to reduce to 2D.
    Color-code the scatter plot by (a) true labels and (b) KMeans
    labels. Visually compare how well KMeans recovers the true classes.

Exercise 5.4: Run t-SNE on Iris with perplexity = [5, 15, 30, 50].
    How does the visualization change? Why does perplexity matter?

Exercise 5.5: Use GaussianMixture with BIC to find the optimal number
    of components for the Iris dataset. Does BIC suggest 3 (the true
    number of species)?

Exercise 5.6: (Advanced) Create a dataset with 3 clusters of varying
    density. Show that DBSCAN fails, and propose a fix (hint: HDBSCAN
    or OPTICS from sklearn.cluster).
"""
