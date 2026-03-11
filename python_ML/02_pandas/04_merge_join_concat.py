"""
Module 2 — Lesson 4: Merge, Join & Concat
===========================================
Real ML pipelines combine data from multiple sources: feature tables,
label files, metadata, external APIs. Pandas provides SQL-like join
operations to combine DataFrames.
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════
# SAMPLE TABLES
# ══════════════════════════════════════════════════════════════════════

users = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "country": ["US", "UK", "US", "DE", "UK"],
})

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104, 105, 106],
    "user_id": [1, 2, 1, 3, 6, 2],   # user_id 6 doesn't exist in users
    "amount": [250, 150, 300, 450, 100, 200],
    "product": ["laptop", "phone", "tablet", "laptop", "headphones", "phone"],
})

scores = pd.DataFrame({
    "user_id": [1, 2, 3],
    "ml_score": [0.92, 0.85, 0.78],
})

print("users:\n", users)
print("\norders:\n", orders)
print("\nscores:\n", scores)


# ══════════════════════════════════════════════════════════════════════
# 1. MERGE (SQL-style joins)
# ══════════════════════════════════════════════════════════════════════
print("\n── Merge (inner join — default) ──")

# Inner join: only matching rows from both tables
inner = pd.merge(orders, users, on="user_id")
print(inner)
# user_id=6 is dropped (not in users), user_id 4,5 not in orders

# Left join: keep ALL rows from left table
print("\n── Left join ──")
left = pd.merge(orders, users, on="user_id", how="left")
print(left)  # user_id=6 has NaN for name/country

# Right join: keep ALL rows from right table
print("\n── Right join ──")
right = pd.merge(orders, users, on="user_id", how="right")
print(right)  # user_id 4,5 have NaN for order columns

# Outer join: keep ALL rows from both
print("\n── Outer join ──")
outer = pd.merge(orders, users, on="user_id", how="outer")
print(outer)

# ── Different column names ───────────────────────────────────────────
df_a = pd.DataFrame({"id_col": [1, 2], "val_a": [10, 20]})
df_b = pd.DataFrame({"key": [1, 2], "val_b": [30, 40]})
merged = pd.merge(df_a, df_b, left_on="id_col", right_on="key")
print("\nDifferent key names:\n", merged)

# ── Handling duplicate columns ───────────────────────────────────────
df1 = pd.DataFrame({"id": [1, 2], "score": [90, 80]})
df2 = pd.DataFrame({"id": [1, 2], "score": [85, 75]})
merged = pd.merge(df1, df2, on="id", suffixes=("_test1", "_test2"))
print("\nWith suffixes:\n", merged)


# ══════════════════════════════════════════════════════════════════════
# 2. MULTIPLE MERGES (chaining)
# ══════════════════════════════════════════════════════════════════════
print("\n── Chained merges ──")

# Combine users + orders + scores
full = (
    users
    .merge(orders, on="user_id", how="left")
    .merge(scores, on="user_id", how="left")
)
print(full)


# ══════════════════════════════════════════════════════════════════════
# 3. CONCAT — Stacking DataFrames
# ══════════════════════════════════════════════════════════════════════
print("\n── Concat ──")

# Vertical stacking (row-wise) — append more samples
batch1 = pd.DataFrame({"feature": [1, 2], "label": [0, 1]})
batch2 = pd.DataFrame({"feature": [3, 4], "label": [1, 0]})
batch3 = pd.DataFrame({"feature": [5, 6], "label": [1, 1]})

combined = pd.concat([batch1, batch2, batch3], ignore_index=True)
print("Vertical concat:")
print(combined)

# Horizontal stacking (column-wise) — add more features
features_a = pd.DataFrame({"feat_1": [10, 20], "feat_2": [30, 40]})
features_b = pd.DataFrame({"feat_3": [50, 60], "feat_4": [70, 80]})
combined_h = pd.concat([features_a, features_b], axis=1)
print("\nHorizontal concat:")
print(combined_h)


# ══════════════════════════════════════════════════════════════════════
# 4. JOIN (index-based merge)
# ══════════════════════════════════════════════════════════════════════
print("\n── Join (index-based) ──")

df_a = pd.DataFrame({"val_a": [1, 2, 3]}, index=["x", "y", "z"])
df_b = pd.DataFrame({"val_b": [10, 20, 30]}, index=["x", "y", "w"])

print(df_a.join(df_b, how="inner"))  # only matching index
print()
print(df_a.join(df_b, how="outer"))  # all indices


# ══════════════════════════════════════════════════════════════════════
# 5. ML PATTERN: Feature Table Assembly
# ══════════════════════════════════════════════════════════════════════
print("\n── ML pattern: feature assembly ──")

# In real ML pipelines, features come from different sources
user_features = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "account_age_days": [365, 730, 180, 90, 1000],
    "total_purchases": [15, 45, 3, 1, 80],
})

behavior_features = pd.DataFrame({
    "user_id": [1, 2, 3, 5],  # user 4 has no behavior data
    "avg_session_min": [12.5, 25.0, 5.2, 30.1],
    "pages_per_session": [8, 15, 3, 20],
})

labels = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "churned": [0, 0, 1, 1, 0],
})

# Assemble: left join to keep all labeled users, fill missing features
dataset = (
    labels
    .merge(user_features, on="user_id", how="left")
    .merge(behavior_features, on="user_id", how="left")
)
print("Assembled dataset:")
print(dataset)
print(f"\nMissing values:\n{dataset.isnull().sum()}")

# Fill missing behavior features with 0 (or median)
dataset = dataset.fillna(0)
print("\nAfter fillna:")
print(dataset)


# ══════════════════════════════════════════════════════════════════════
# 6. MERGE DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════
print("\n── Merge diagnostics ──")

# validate: check merge cardinality
try:
    pd.merge(users, orders, on="user_id", validate="one_to_one")
except Exception as e:
    print(f"Validation error: {e}")

# indicator: shows where each row came from
result = pd.merge(users, orders, on="user_id", how="outer", indicator=True)
print("\nWith indicator:")
print(result[["user_id", "name", "order_id", "_merge"]])
print("\nMerge counts:")
print(result["_merge"].value_counts())


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 4.1: Create three DataFrames representing:
    - train_features (1000 rows × 5 features, indexed by "sample_id")
    - train_labels (1000 rows, "sample_id" + "label")
    - metadata (1000 rows, "sample_id" + "source", "timestamp")
    Merge them all into a single training DataFrame.

Exercise 4.2: Simulate a scenario where you have:
    - predictions from 3 different models (each a DataFrame with "id" and "pred")
    - Merge them side by side to create an ensemble DataFrame
    - Add a column "ensemble_pred" = mean of the 3 predictions

Exercise 4.3: Use the merge indicator to find:
    - Users who placed orders (both)
    - Users who never ordered (left_only)
    - Orders from unknown users (right_only)

Exercise 4.4: Load two CSV files (or create them) with overlapping columns.
    Practice concat with ignore_index=True. What happens if columns
    don't match? (Answer: NaN for missing columns)
"""
