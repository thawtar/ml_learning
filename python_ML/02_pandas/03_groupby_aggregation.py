"""
Module 2 — Lesson 3: GroupBy & Aggregation
============================================
split-apply-combine is one of the most powerful patterns in data analysis.
It's how you compute per-group statistics — essential for feature
engineering and EDA in ML.
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════
# SAMPLE DATASET
# ══════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)

df = pd.DataFrame({
    "department": rng.choice(["Engineering", "Sales", "Marketing"], 50),
    "role": rng.choice(["Junior", "Senior", "Lead"], 50),
    "salary": rng.normal(75000, 15000, 50).round(2),
    "experience": rng.integers(1, 20, 50),
    "performance": rng.uniform(1, 5, 50).round(2),
})
print(df.head(10))


# ══════════════════════════════════════════════════════════════════════
# 1. BASIC GROUPBY
# ══════════════════════════════════════════════════════════════════════
print("\n── Basic groupby ──")

# The pattern: df.groupby(key).aggregation()
grouped = df.groupby("department")
print(type(grouped))   # DataFrameGroupBy — a lazy object

# Mean salary per department
print("\nMean salary by department:")
print(grouped["salary"].mean().round(2))

# Multiple aggregations
print("\nSalary stats by department:")
print(grouped["salary"].agg(["mean", "median", "std", "min", "max"]).round(2))

# Size (count of rows per group)
print("\nGroup sizes:")
print(grouped.size())


# ══════════════════════════════════════════════════════════════════════
# 2. MULTI-COLUMN GROUPBY
# ══════════════════════════════════════════════════════════════════════
print("\n── Multi-column groupby ──")

result = df.groupby(["department", "role"])["salary"].mean().round(2)
print("Mean salary by dept + role:")
print(result)

# Unstack — pivot the inner index level to columns
print("\nUnstacked:")
print(result.unstack(fill_value=0))


# ══════════════════════════════════════════════════════════════════════
# 3. NAMED AGGREGATION (.agg with named outputs)
# ══════════════════════════════════════════════════════════════════════
print("\n── Named aggregation ──")

summary = df.groupby("department").agg(
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    avg_perf=("performance", "mean"),
    headcount=("salary", "count"),
    avg_exp=("experience", "mean"),
).round(2)
print(summary)


# ══════════════════════════════════════════════════════════════════════
# 4. TRANSFORM — Group-Level Ops That Preserve Shape
# ══════════════════════════════════════════════════════════════════════
print("\n── Transform ──")

# transform returns a Series with the SAME index as the original DataFrame
# (unlike agg, which collapses groups)

# Example: standardize salary within each department
df["salary_zscore"] = df.groupby("department")["salary"].transform(
    lambda x: (x - x.mean()) / x.std()
).round(3)

print(df[["department", "salary", "salary_zscore"]].head(10))

# Example: percentage of department salary
df["salary_pct"] = df.groupby("department")["salary"].transform(
    lambda x: x / x.sum()
).round(4)

# Example: flag above-average performers per department
dept_mean_perf = df.groupby("department")["performance"].transform("mean")
df["above_avg"] = df["performance"] > dept_mean_perf
print("\nAbove-average performers per dept:")
print(df.groupby("department")["above_avg"].sum())


# ══════════════════════════════════════════════════════════════════════
# 5. FILTER — Keep Only Groups Meeting a Condition
# ══════════════════════════════════════════════════════════════════════
print("\n── Filter ──")

# Keep only departments with more than 15 employees
large_depts = df.groupby("department").filter(lambda g: len(g) > 15)
print(f"Rows after filter: {len(large_depts)} (from {len(df)})")
print("Remaining departments:", large_depts["department"].unique())


# ══════════════════════════════════════════════════════════════════════
# 6. PIVOT TABLES
# ══════════════════════════════════════════════════════════════════════
print("\n── Pivot tables ──")

pivot = df.pivot_table(
    values="salary",
    index="department",
    columns="role",
    aggfunc="mean",
    fill_value=0,
).round(2)
print(pivot)

# Multiple aggregations
pivot2 = df.pivot_table(
    values="salary",
    index="department",
    aggfunc=["mean", "count"],
).round(2)
print("\nMulti-agg pivot:")
print(pivot2)


# ══════════════════════════════════════════════════════════════════════
# 7. VALUE_COUNTS & CROSSTAB
# ══════════════════════════════════════════════════════════════════════
print("\n── value_counts & crosstab ──")

print("Department distribution:")
print(df["department"].value_counts())

print("\nNormalized (proportions):")
print(df["department"].value_counts(normalize=True).round(3))

# Crosstab — frequency table of two categorical variables
print("\nCrosstab (dept × role):")
print(pd.crosstab(df["department"], df["role"]))

print("\nCrosstab (proportions by row):")
print(pd.crosstab(df["department"], df["role"], normalize="index").round(3))


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 3.1: Using the Titanic dataset, compute:
    - Survival rate by passenger class (Pclass)
    - Average fare by class and sex
    - Survival rate by age group (create bins: child <18, adult 18-60, senior >60)

Exercise 3.2: Use transform to add a column "fare_pct_of_class" that shows
    each passenger's fare as a percentage of their class's total fare.

Exercise 3.3: Use groupby + agg to create an ML feature engineering summary:
    For each department, compute: mean, std, median, min, max of salary.
    These are the kind of grouped statistics you'd use as features.

Exercise 3.4: Create a pivot table showing:
    - Rows: department
    - Columns: role
    - Values: median performance score
    Which department + role combination has the highest median performance?

Exercise 3.5: Use .filter() to keep only departments where the average
    experience is above 8 years. How many rows remain?
"""
