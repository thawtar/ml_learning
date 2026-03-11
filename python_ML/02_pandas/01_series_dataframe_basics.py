"""
Module 2 — Lesson 1: Pandas Series & DataFrame Basics
=======================================================
Pandas is the standard for tabular data in Python ML pipelines.
It wraps NumPy arrays with labels (index + column names) and provides
powerful I/O, alignment, and missing-data handling.

Key objects:
  - Series: 1D labeled array (like a column)
  - DataFrame: 2D labeled table (like a spreadsheet / SQL table)
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════
# 1. SERIES
# ══════════════════════════════════════════════════════════════════════
print("── Series ──")

# From a list
s = pd.Series([10, 20, 30, 40], name="scores")
print(s)
print(f"dtype: {s.dtype}, shape: {s.shape}, name: {s.name}\n")

# From a dict (keys become the index)
temps = pd.Series({"Mon": 72, "Tue": 75, "Wed": 68, "Thu": 80, "Fri": 77})
print("Temps:\n", temps)
print("Wed temp:", temps["Wed"])       # label-based access
print("First 3:", temps[:3].values)    # slicing works like NumPy

# Vectorized operations (just like NumPy)
print("\nCelsius:", ((temps - 32) * 5 / 9).round(1).values)

# Boolean indexing
print("Hot days (>75):\n", temps[temps > 75])


# ══════════════════════════════════════════════════════════════════════
# 2. DATAFRAME CREATION
# ══════════════════════════════════════════════════════════════════════
print("\n── DataFrame creation ──")

# From a dict of lists
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "age": [28, 34, 22, 31],
    "salary": [70000, 85000, 55000, 92000],
    "department": ["Engineering", "Sales", "Engineering", "Marketing"],
})
print(df)
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes}")

# From a NumPy array
data = np.random.randn(5, 3)
df_np = pd.DataFrame(data, columns=["feature_1", "feature_2", "feature_3"])
print("\nFrom NumPy:\n", df_np.round(3))

# From a list of dicts (common when parsing JSON APIs)
records = [
    {"id": 1, "label": "cat", "confidence": 0.95},
    {"id": 2, "label": "dog", "confidence": 0.87},
    {"id": 3, "label": "cat", "confidence": 0.72},
]
df_records = pd.DataFrame(records)
print("\nFrom records:\n", df_records)


# ══════════════════════════════════════════════════════════════════════
# 3. INSPECTION — The First Thing You Do With Any Dataset
# ══════════════════════════════════════════════════════════════════════
print("\n── Inspection ──")

# Create a richer dataset
rng = np.random.default_rng(42)
n = 200
df = pd.DataFrame({
    "age": rng.integers(18, 65, n),
    "income": rng.normal(50000, 15000, n).round(2),
    "credit_score": rng.integers(300, 850, n),
    "approved": rng.choice([True, False], n, p=[0.6, 0.4]),
    "category": rng.choice(["A", "B", "C"], n),
})

print(df.head())            # first 5 rows
print(df.tail(3))           # last 3 rows
print(df.shape)             # (200, 5)
print(df.dtypes)            # column types
print()
print(df.info())            # summary: dtypes, non-null counts, memory
print()
print(df.describe())        # statistics for numeric columns
print()
print(df.describe(include="object"))  # statistics for string/categorical


# ══════════════════════════════════════════════════════════════════════
# 4. INDEXING: loc vs iloc
# ══════════════════════════════════════════════════════════════════════
print("\n── Indexing ──")

df_small = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "score": [90, 85, 92],
    "grade": ["A", "B", "A"],
}, index=["s1", "s2", "s3"])

print(df_small)

# .loc — LABEL-based indexing (inclusive on both ends)
print("\nloc['s1']:\n", df_small.loc["s1"])              # Series (one row)
print("loc['s1','score']:", df_small.loc["s1", "score"])  # single value
print("loc['s1':'s2']:\n", df_small.loc["s1":"s2"])      # s1 AND s2 (inclusive!)
print("loc[:, 'score']:\n", df_small.loc[:, "score"])    # all rows, one column

# .iloc — INTEGER-based indexing (exclusive on end, like Python)
print("\niloc[0]:\n", df_small.iloc[0])                   # first row
print("iloc[0:2]:\n", df_small.iloc[0:2])                # rows 0,1 (not 2!)
print("iloc[:, 1]:\n", df_small.iloc[:, 1])              # second column

# ⚠ CRITICAL RULE:
# .loc = LABEL-based (uses the index values)
# .iloc = INTEGER POSITION-based (uses 0,1,2... like a list)
# Mixing them up is the #1 Pandas bug.

# Column access shortcuts
print("\ndf['score']:\n", df_small["score"])   # single column → Series
print("df[['name','score']]:\n", df_small[["name", "score"]])  # multiple → DataFrame

# Boolean filtering
print("\nHigh scorers:\n", df_small[df_small["score"] >= 90])

# Chaining: filter rows AND select columns
print("\nNames with A grade:", df_small.loc[df_small["grade"] == "A", "name"].values)


# ══════════════════════════════════════════════════════════════════════
# 5. ADDING, MODIFYING, REMOVING COLUMNS
# ══════════════════════════════════════════════════════════════════════
print("\n── Column operations ──")

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "hours_worked": [160, 180, 150],
    "hourly_rate": [50, 45, 55],
})

# Add a new column (vectorized)
df["monthly_pay"] = df["hours_worked"] * df["hourly_rate"]
print(df)

# Conditional column
df["senior"] = df["monthly_pay"] > 7500
print(df)

# Using assign (returns new DataFrame — method chaining friendly)
df2 = df.assign(
    tax=lambda d: d["monthly_pay"] * 0.2,
    net_pay=lambda d: d["monthly_pay"] * 0.8,
)
print("\nWith assign:\n", df2)

# Drop columns
df = df.drop(columns=["senior"])
print("\nAfter drop:\n", df)

# Rename columns
df = df.rename(columns={"hours_worked": "hours", "hourly_rate": "rate"})
print("\nRenamed:\n", df.columns.tolist())


# ══════════════════════════════════════════════════════════════════════
# 6. READING & WRITING DATA
# ══════════════════════════════════════════════════════════════════════
print("\n── I/O ──")

# CSV (most common)
# df.to_csv("output.csv", index=False)
# df = pd.read_csv("data.csv")

# Parquet (faster, smaller, preserves types — preferred for ML)
# df.to_parquet("output.parquet")
# df = pd.read_parquet("data.parquet")

# Excel
# df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# SQL
# df = pd.read_sql("SELECT * FROM users", connection)

# JSON
# df = pd.read_json("data.json")

# From clipboard (quick exploration)
# df = pd.read_clipboard()

print("Common I/O functions: read_csv, read_parquet, read_excel, read_sql, read_json")
print("Pandas supports 20+ file formats out of the box.")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 1.1: Create a DataFrame with 1000 rows representing ML experiments:
    - "learning_rate": random uniform in [0.0001, 0.1]
    - "batch_size": random choice from [16, 32, 64, 128, 256]
    - "accuracy": random normal(0.85, 0.05), clipped to [0, 1]
    - "loss": random normal(0.3, 0.1), clipped to [0, inf)
    Use .describe() to summarize. Filter experiments with accuracy > 0.9.

Exercise 1.2: Using the DataFrame from 1.1:
    - Select the top 10 experiments by accuracy
    - Select all experiments where batch_size == 64 AND accuracy > 0.88
    - Use .loc to select rows 5-10 and columns "learning_rate" and "accuracy"

Exercise 1.3: Create a DataFrame from this dict and set "id" as the index:
    {"id": [101, 102, 103], "model": ["RF", "XGB", "LR"], "f1": [0.87, 0.91, 0.83]}
    Practice using both .loc (label) and .iloc (position) to access data.

Exercise 1.4: Read a CSV from a URL:
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    Inspect with .head(), .info(), .describe(), .value_counts() on species.
"""
