"""
Module 3 — Lesson 2: Statistical Plots (Seaborn + Matplotlib)
==============================================================
Seaborn is built on Matplotlib and provides high-level functions
for statistical visualization. It integrates with Pandas DataFrames
and automatically handles group-level aesthetics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try importing seaborn — provide guidance if not installed
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAS_SEABORN = True
except ImportError:
    print("⚠ seaborn not installed. Run: pip install seaborn")
    HAS_SEABORN = False


# ══════════════════════════════════════════════════════════════════════
# SAMPLE DATASET
# ══════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)
n = 300

df = pd.DataFrame({
    "feature_1": rng.normal(0, 1, n),
    "feature_2": rng.normal(0, 1, n),
    "feature_3": rng.exponential(2, n),
    "target": rng.choice(["Class A", "Class B", "Class C"], n, p=[0.5, 0.3, 0.2]),
    "score": rng.normal(75, 10, n).round(1),
    "age": rng.integers(20, 60, n),
})
df["feature_2"] = df["feature_1"] * 0.6 + df["feature_2"] * 0.4  # correlated

if not HAS_SEABORN:
    print("Skipping seaborn plots. Install with: pip install seaborn")
    exit()


# ══════════════════════════════════════════════════════════════════════
# 1. DISTRIBUTION PLOTS
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram + KDE (kernel density estimate)
sns.histplot(df["score"], bins=30, kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Histogram + KDE")

# KDE by group
sns.kdeplot(data=df, x="score", hue="target", ax=axes[0, 1], fill=True, alpha=0.3)
axes[0, 1].set_title("KDE by Class")

# Box plot — shows median, IQR, outliers
sns.boxplot(data=df, x="target", y="score", ax=axes[1, 0], palette="Set2")
axes[1, 0].set_title("Box Plot by Class")

# Violin plot — box plot + KDE shape
sns.violinplot(data=df, x="target", y="score", ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title("Violin Plot by Class")

fig.tight_layout()
fig.savefig("stat_01_distributions.png", dpi=150, bbox_inches="tight")
print("Saved: stat_01_distributions.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 2. RELATIONSHIP PLOTS
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Scatter with hue (grouping)
sns.scatterplot(data=df, x="feature_1", y="feature_2", hue="target",
                alpha=0.6, ax=axes[0])
axes[0].set_title("Scatter by Class")

# Regression plot (scatter + fitted line)
sns.regplot(data=df, x="feature_1", y="feature_2", ax=axes[1],
            scatter_kws={"alpha": 0.4})
axes[1].set_title("Regression Plot")

# Hexbin for dense data
axes[2].hexbin(df["feature_1"], df["feature_2"], gridsize=20, cmap="YlGnBu")
axes[2].set_title("Hexbin Density")

fig.tight_layout()
fig.savefig("stat_02_relationships.png", dpi=150, bbox_inches="tight")
print("Saved: stat_02_relationships.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 3. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════

# Select numeric columns
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix")

fig.savefig("stat_03_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: stat_03_heatmap.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 4. PAIR PLOT — All Pairwise Relationships
# ══════════════════════════════════════════════════════════════════════

# pairplot creates a grid of scatter plots and distributions
g = sns.pairplot(df[["feature_1", "feature_2", "score", "target"]],
                 hue="target", diag_kind="kde", plot_kws={"alpha": 0.5})
g.figure.suptitle("Pair Plot", y=1.02)
g.savefig("stat_04_pairplot.png", dpi=150, bbox_inches="tight")
print("Saved: stat_04_pairplot.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 5. CATEGORICAL PLOTS
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Count plot (bar chart of counts)
sns.countplot(data=df, x="target", palette="Set2", ax=axes[0])
axes[0].set_title("Class Distribution")

# Strip plot (individual points by category)
sns.stripplot(data=df, x="target", y="score", alpha=0.4, jitter=True, ax=axes[1])
axes[1].set_title("Strip Plot")

# Swarm plot (non-overlapping points)
small_df = df.sample(50, random_state=42)
sns.swarmplot(data=small_df, x="target", y="score", ax=axes[2], palette="Set1")
axes[2].set_title("Swarm Plot (50 samples)")

fig.tight_layout()
fig.savefig("stat_05_categorical.png", dpi=150, bbox_inches="tight")
print("Saved: stat_05_categorical.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 6. FACETING — Small Multiples
# ══════════════════════════════════════════════════════════════════════

g = sns.FacetGrid(df, col="target", height=4, aspect=1)
g.map_dataframe(sns.histplot, x="score", bins=20, color="steelblue")
g.set_titles("{col_name}")
g.figure.suptitle("Score Distribution by Class", y=1.02)
g.savefig("stat_06_facet.png", dpi=150, bbox_inches="tight")
print("Saved: stat_06_facet.png")
plt.close()

print("\nAll statistical plots saved.")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 2.1: Load the Iris dataset (sns.load_dataset("iris")).
    Create: pair plot colored by species, correlation heatmap,
    and box plots of each feature by species.

Exercise 2.2: Generate two clusters of 2D data (Gaussian blobs).
    Create a scatter plot with:
    - Points colored by cluster label
    - Cluster centers marked with 'X'
    - Confidence ellipses at 1σ and 2σ

Exercise 2.3: Create a figure with 4 subplots showing the same
    continuous variable (score) as:
    a) Histogram  b) KDE  c) Box plot  d) Violin plot

Exercise 2.4: Generate a correlation matrix for 8 random features.
    Create a heatmap that only shows the lower triangle
    (hint: use np.triu to mask the upper triangle).
"""
