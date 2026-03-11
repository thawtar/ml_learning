"""
Module 2 — Lesson 2: Data Cleaning with Pandas
================================================
Real-world ML data is messy. Before modeling, you spend 60-80% of
your time cleaning data. This lesson covers the essential operations.
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════
# CREATE A MESSY DATASET
# ══════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", None, "Eve", "alice", "Bob", "Diana"],
    "age": [28, 34, None, 22, 45, 28, 34, -5],
    "salary": [70000, 85000, 55000, None, 92000, 70000, 85000, 45000],
    "department": ["Engineering", "SALES", "engineering", "Marketing", None, "Engineering", "Sales", "marketing"],
    "start_date": ["2020-01-15", "2019-06-01", "2021-03-10", "2022-invalid", "2018-11-20", "2020-01-15", "2019-06-01", "2023-01-01"],
    "rating": ["4.5", "3.8", "4.2", "N/A", "4.9", "4.5", "3.8", "invalid"],
})
print("Original messy data:")
print(df)
print()


# ══════════════════════════════════════════════════════════════════════
# 1. MISSING VALUES (NaN / None)
# ══════════════════════════════════════════════════════════════════════
print("── Missing values ──")

# Detect missing
print("Is null:\n", df.isnull().sum())   # count NaN per column
print("Total nulls:", df.isnull().sum().sum())

# Drop rows with ANY missing value
df_dropped = df.dropna()
print(f"\nAfter dropna: {len(df_dropped)} rows (from {len(df)})")

# Drop rows where specific columns have NaN
df_partial = df.dropna(subset=["name", "salary"])
print(f"After dropna(subset): {len(df_partial)} rows")

# Fill missing values
df_filled = df.copy()
df_filled["age"] = df_filled["age"].fillna(df_filled["age"].median())
df_filled["salary"] = df_filled["salary"].fillna(df_filled["salary"].mean())
df_filled["department"] = df_filled["department"].fillna("Unknown")
print("\nAfter fillna:\n", df_filled[["name", "age", "salary", "department"]])

# Forward fill / backward fill (for time series)
ts = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
print("\nffill:", ts.ffill().values)   # [1, 1, 1, 4, 4, 6]
print("bfill:", ts.bfill().values)     # [1, 4, 4, 4, 6, 6]

# Interpolate (linear interpolation)
print("interp:", ts.interpolate().values)  # [1, 2, 3, 4, 5, 6]


# ══════════════════════════════════════════════════════════════════════
# 2. DUPLICATES
# ══════════════════════════════════════════════════════════════════════
print("\n── Duplicates ──")

# Check duplicates
print("Duplicate rows:\n", df[df.duplicated(keep=False)])
print("Duplicate count:", df.duplicated().sum())

# Drop duplicates
df_deduped = df.drop_duplicates()
print(f"After dedup: {len(df_deduped)} rows (from {len(df)})")

# Deduplicate based on specific columns
df_name_dedup = df.drop_duplicates(subset=["name"], keep="first")
print(f"Unique names: {len(df_name_dedup)} rows")


# ══════════════════════════════════════════════════════════════════════
# 3. TYPE CONVERSION
# ══════════════════════════════════════════════════════════════════════
print("\n── Type conversion ──")

df_clean = df.copy()

# Convert rating to numeric (coerce errors to NaN)
df_clean["rating"] = pd.to_numeric(df_clean["rating"], errors="coerce")
print("Rating dtype:", df_clean["rating"].dtype)
print("Rating values:", df_clean["rating"].values)

# Convert dates (coerce invalid dates to NaT — Not a Time)
df_clean["start_date"] = pd.to_datetime(df_clean["start_date"], errors="coerce")
print("\nDate dtype:", df_clean["start_date"].dtype)
print(df_clean["start_date"])

# Convert to category (saves memory for low-cardinality strings)
df_clean["department"] = df_clean["department"].astype("category")
print(f"\nDepartment dtype: {df_clean['department'].dtype}")

# Nullable integer type (regular int can't hold NaN)
df_clean["age"] = df_clean["age"].astype("Int64")  # capital I = nullable
print(f"Age dtype: {df_clean['age'].dtype}")


# ══════════════════════════════════════════════════════════════════════
# 4. STRING OPERATIONS
# ══════════════════════════════════════════════════════════════════════
print("\n── String operations ──")

# Access string methods via .str accessor
df_str = df.copy()

# Standardize case
df_str["name"] = df_str["name"].str.strip().str.title()
df_str["department"] = df_str["department"].str.strip().str.lower()
print(df_str[["name", "department"]])

# String matching
print("\nNames containing 'li':", df_str["name"].str.contains("li", na=False).values)

# Extract patterns with regex
emails = pd.Series(["alice@corp.com", "bob@uni.edu", "charlie@corp.com"])
domains = emails.str.extract(r"@(\w+\.\w+)")
print("\nDomains:\n", domains)

# Replace
df_str["department"] = df_str["department"].str.replace("engineering", "eng")
print("\nAfter replace:", df_str["department"].values)

# Split
names = pd.Series(["Alice Smith", "Bob Johnson", "Charlie Brown"])
split = names.str.split(" ", expand=True)
split.columns = ["first", "last"]
print("\nSplit names:\n", split)


# ══════════════════════════════════════════════════════════════════════
# 5. OUTLIER DETECTION & HANDLING
# ══════════════════════════════════════════════════════════════════════
print("\n── Outliers ──")

data = pd.DataFrame({
    "value": [10, 12, 11, 13, 100, 9, 11, 10, -50, 12],
})

# Method 1: IQR (Interquartile Range)
Q1 = data["value"].quantile(0.25)
Q3 = data["value"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers_iqr = data[(data["value"] < lower) | (data["value"] > upper)]
print(f"IQR bounds: [{lower:.1f}, {upper:.1f}]")
print(f"IQR outliers:\n{outliers_iqr}")

# Method 2: Z-score
z_scores = (data["value"] - data["value"].mean()) / data["value"].std()
outliers_z = data[z_scores.abs() > 2]
print(f"\nZ-score outliers (|z|>2):\n{outliers_z}")

# Clip outliers
data["clipped"] = data["value"].clip(lower=lower, upper=upper)
print(f"\nClipped:\n{data}")


# ══════════════════════════════════════════════════════════════════════
# 6. APPLY & MAP — Custom Transformations
# ══════════════════════════════════════════════════════════════════════
print("\n── apply & map ──")

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "score": [85, 92, 78],
})

# map — element-wise on a Series
df["grade"] = df["score"].map(lambda x: "Pass" if x >= 80 else "Fail")
print(df)

# map with a dict (great for label encoding)
grade_map = {"Pass": 1, "Fail": 0}
df["grade_num"] = df["grade"].map(grade_map)
print(df)

# apply — row-wise or column-wise on a DataFrame
df["summary"] = df.apply(lambda row: f"{row['name']}: {row['grade']}", axis=1)
print(df)

# ⚠ Performance: avoid apply when a vectorized operation exists.
# apply is a Python loop internally — use it only for complex logic.


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 2.1: Load the Titanic dataset:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    - How many missing values per column?
    - Fill 'Age' with the median, 'Embarked' with the mode
    - Drop the 'Cabin' column (too many missing values)

Exercise 2.2: Given a column of messy strings:
    cities = pd.Series(["  New York ", "new york", "NEW YORK", "  Boston", "boston "])
    Standardize them to lowercase, stripped, title case. Count unique values.

Exercise 2.3: Create a DataFrame with intentional outliers in a numerical
    column. Implement both IQR and Z-score outlier detection. Compare which
    method catches more outliers.

Exercise 2.4: Write a cleaning pipeline function that:
    - Drops duplicates
    - Fills numeric NaN with median
    - Fills categorical NaN with mode
    - Converts string columns to lowercase
    - Returns the cleaned DataFrame
    Apply it to the Titanic dataset.
"""
