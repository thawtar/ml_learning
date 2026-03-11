"""
Module 2 — Lesson 5: Feature Engineering with Pandas
======================================================
Feature engineering is the art of transforming raw data into useful
model inputs. Good features > complex models. This lesson covers
the most common Pandas-based feature engineering techniques.
"""

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════
# SAMPLE DATASET
# ══════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(42)

n = 200
df = pd.DataFrame({
    "user_id": range(1, n + 1),
    "age": rng.integers(18, 70, n),
    "income": rng.lognormal(11, 0.5, n).round(2),
    "city": rng.choice(["New York", "San Francisco", "Chicago", "Austin", "Seattle"], n),
    "signup_date": pd.date_range("2020-01-01", periods=n, freq="2D"),
    "last_login": pd.date_range("2020-01-01", periods=n, freq="2D") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    "purchases": rng.poisson(5, n),
    "product_category": rng.choice(["Electronics", "Clothing", "Food", "Books"], n),
    "satisfaction": rng.uniform(1, 10, n).round(1),
})
print(df.head())


# ══════════════════════════════════════════════════════════════════════
# 1. BINNING / DISCRETIZATION
# ══════════════════════════════════════════════════════════════════════
print("\n── Binning ──")

# Equal-width bins
df["age_group"] = pd.cut(df["age"], bins=[0, 25, 35, 50, 100],
                          labels=["Young", "Adult", "Middle", "Senior"])
print(df[["age", "age_group"]].head(10))
print("\nAge group distribution:")
print(df["age_group"].value_counts().sort_index())

# Quantile bins (equal-frequency)
df["income_quartile"] = pd.qcut(df["income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
print("\nIncome quartile distribution:")
print(df["income_quartile"].value_counts().sort_index())


# ══════════════════════════════════════════════════════════════════════
# 2. ENCODING CATEGORICAL VARIABLES
# ══════════════════════════════════════════════════════════════════════
print("\n── Categorical encoding ──")

# Label encoding (ordinal — implies order)
# Use when categories have a natural order
size_map = {"Small": 0, "Medium": 1, "Large": 2}
# df["size_encoded"] = df["size"].map(size_map)

# One-hot encoding (nominal — no order implied)
city_dummies = pd.get_dummies(df["city"], prefix="city", dtype=int)
print("One-hot encoded cities:")
print(city_dummies.head())

# Add to dataframe (drop original)
df_encoded = pd.concat([df, city_dummies], axis=1)
print(f"\nShape after one-hot: {df_encoded.shape}")

# ⚠ Drop one dummy to avoid multicollinearity (for linear models)
city_dummies_k1 = pd.get_dummies(df["city"], prefix="city", drop_first=True, dtype=int)
print(f"With drop_first: {city_dummies_k1.columns.tolist()}")

# Frequency encoding (replace category with its frequency)
freq = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq)
print("\nFrequency encoding:")
print(df[["city", "city_freq"]].drop_duplicates().sort_values("city"))

# Target encoding (replace category with mean of target — careful of leakage!)
# Typically done within cross-validation folds


# ══════════════════════════════════════════════════════════════════════
# 3. DATETIME FEATURES
# ══════════════════════════════════════════════════════════════════════
print("\n── Datetime features ──")

# Extract components via .dt accessor
df["signup_year"] = df["signup_date"].dt.year
df["signup_month"] = df["signup_date"].dt.month
df["signup_dow"] = df["signup_date"].dt.dayofweek    # 0=Monday
df["signup_quarter"] = df["signup_date"].dt.quarter

print(df[["signup_date", "signup_year", "signup_month", "signup_dow"]].head())

# Time since an event (recency feature)
reference_date = pd.Timestamp("2024-01-01")
df["days_since_signup"] = (reference_date - df["signup_date"]).dt.days
df["days_since_login"] = (reference_date - df["last_login"]).dt.days
print("\nRecency features:")
print(df[["signup_date", "days_since_signup", "last_login", "days_since_login"]].head())

# Cyclical encoding (for periodic features like month, hour, day_of_week)
df["month_sin"] = np.sin(2 * np.pi * df["signup_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["signup_month"] / 12)
print("\nCyclical month encoding:")
print(df[["signup_month", "month_sin", "month_cos"]].head(8).round(3))


# ══════════════════════════════════════════════════════════════════════
# 4. WINDOW / ROLLING FEATURES
# ══════════════════════════════════════════════════════════════════════
print("\n── Rolling / window features ──")

# Simulate time series data
ts = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=30, freq="D"),
    "sales": rng.poisson(100, 30) + np.sin(np.arange(30) * 0.3) * 20,
}).set_index("date")

# Rolling mean (moving average)
ts["rolling_7d"] = ts["sales"].rolling(window=7).mean()

# Rolling std (volatility)
ts["rolling_std_7d"] = ts["sales"].rolling(window=7).std()

# Expanding mean (cumulative average)
ts["expanding_mean"] = ts["sales"].expanding().mean()

# Lag features (previous values)
ts["lag_1"] = ts["sales"].shift(1)    # yesterday's sales
ts["lag_7"] = ts["sales"].shift(7)    # last week's sales

# Percentage change
ts["pct_change"] = ts["sales"].pct_change()

print(ts.round(2).head(10))


# ══════════════════════════════════════════════════════════════════════
# 5. INTERACTION & POLYNOMIAL FEATURES
# ══════════════════════════════════════════════════════════════════════
print("\n── Interaction features ──")

# Simple interactions
df["income_per_purchase"] = df["income"] / (df["purchases"] + 1)  # +1 to avoid div by 0
df["age_times_income"] = df["age"] * df["income"]

print(df[["age", "income", "purchases", "income_per_purchase"]].head())

# Log transform (for skewed distributions like income)
df["log_income"] = np.log1p(df["income"])   # log1p = log(1 + x), handles 0

# Polynomial features
df["age_squared"] = df["age"] ** 2

# Ratio features
df["satisfaction_per_purchase"] = df["satisfaction"] / (df["purchases"] + 1)

print("\nTransformed features:")
print(df[["income", "log_income", "age", "age_squared"]].head())


# ══════════════════════════════════════════════════════════════════════
# 6. GROUP-BASED FEATURES (Aggregation Features)
# ══════════════════════════════════════════════════════════════════════
print("\n── Group-based features ──")

# City-level statistics as features
city_stats = df.groupby("city").agg(
    city_avg_income=("income", "mean"),
    city_avg_age=("age", "mean"),
    city_total_purchases=("purchases", "sum"),
).round(2)
print("City stats:")
print(city_stats)

# Merge back to get per-user city-level features
df = df.merge(city_stats, on="city", how="left")

# Relative features: how does this user compare to their city?
df["income_vs_city"] = df["income"] / df["city_avg_income"]

print("\nRelative features:")
print(df[["user_id", "city", "income", "city_avg_income", "income_vs_city"]].head())


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 5.1: Load the Titanic dataset. Engineer these features:
    - "title": extract from Name (Mr., Mrs., Miss., etc.)
    - "family_size": SibSp + Parch + 1
    - "is_alone": family_size == 1
    - "fare_per_person": Fare / family_size
    - One-hot encode "Embarked"

Exercise 5.2: Create a synthetic transaction dataset with timestamps.
    Engineer rolling features: 7-day rolling sum of purchases, 
    30-day rolling mean of purchase amount.

Exercise 5.3: Given a dataset with a "price" column, create:
    - Log-transformed price
    - Quantile-binned price (4 bins)
    - Price relative to category mean (using groupby + transform)
    Compare distributions using .describe().

Exercise 5.4: Implement a complete feature engineering pipeline as a function:
    def engineer_features(df):
        # 1. Handle dates → extract components + recency
        # 2. Encode categoricals → one-hot for low cardinality, frequency for high
        # 3. Create interactions → ratios, products
        # 4. Add group statistics
        return df_features
"""
