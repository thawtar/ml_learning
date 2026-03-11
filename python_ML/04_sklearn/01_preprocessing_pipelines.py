"""
Module 4 — Lesson 1: Preprocessing & Pipelines
================================================
Scikit-learn's Pipeline and ColumnTransformer are the backbone of
production-quality ML code. They prevent data leakage, make code
reproducible, and simplify deployment.

Golden rule: NEVER fit transformers on test data.
Pipeline ensures this automatically.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
    PolynomialFeatures, FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ══════════════════════════════════════════════════════════════════════
# 1. SCALERS — When & Why
# ══════════════════════════════════════════════════════════════════════
print("── Scalers ──")

rng = np.random.default_rng(42)
X = rng.normal(loc=[100, 0.5, 1000], scale=[15, 0.1, 200], size=(100, 3))
print(f"Original: mean={X.mean(axis=0).round(2)}, std={X.std(axis=0).round(2)}")

# StandardScaler: (x - mean) / std → mean=0, std=1
# Use for: linear models, SVM, neural networks
ss = StandardScaler()
X_standard = ss.fit_transform(X)
print(f"Standard: mean={X_standard.mean(axis=0).round(4)}, std={X_standard.std(axis=0).round(4)}")

# MinMaxScaler: (x - min) / (max - min) → [0, 1]
# Use for: neural networks, algorithms sensitive to magnitude
mm = MinMaxScaler()
X_minmax = mm.fit_transform(X)
print(f"MinMax:   min={X_minmax.min(axis=0).round(4)}, max={X_minmax.max(axis=0).round(4)}")

# RobustScaler: (x - median) / IQR → robust to outliers
# Use for: data with outliers
rs = RobustScaler()
X_robust = rs.fit_transform(X)
print(f"Robust:   median≈{X_robust.mean(axis=0).round(2)}")

# ⚠ Tree-based models (RF, XGBoost) do NOT need scaling
# ⚠ ALWAYS fit on train data, transform both train and test:
#     scaler.fit(X_train)
#     X_train_scaled = scaler.transform(X_train)
#     X_test_scaled = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════════════════
# 2. ENCODING CATEGORICAL FEATURES
# ══════════════════════════════════════════════════════════════════════
print("\n── Encoding ──")

# OneHotEncoder — for nominal (unordered) categories
ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
categories = np.array([["red"], ["blue"], ["green"], ["red"]])
encoded = ohe.fit_transform(categories)
print("One-hot (drop_first):")
print(f"  Categories: {ohe.categories_}")
print(f"  Encoded:\n{encoded}")

# OrdinalEncoder — for ordered categories
oe = OrdinalEncoder(categories=[["low", "medium", "high"]])
ordered = np.array([["medium"], ["high"], ["low"], ["high"]])
print(f"\nOrdinal: {oe.fit_transform(ordered).ravel()}")


# ══════════════════════════════════════════════════════════════════════
# 3. IMPUTATION
# ══════════════════════════════════════════════════════════════════════
print("\n── Imputation ──")

X_missing = np.array([[1, 2, np.nan],
                       [4, np.nan, 6],
                       [7, 8, 9],
                       [np.nan, 11, 12]])

# Mean imputation
imp_mean = SimpleImputer(strategy="mean")
print("Mean imputed:\n", imp_mean.fit_transform(X_missing).round(2))

# Median imputation (better for skewed data)
imp_median = SimpleImputer(strategy="median")
print("Median imputed:\n", imp_median.fit_transform(X_missing).round(2))

# Constant imputation
imp_const = SimpleImputer(strategy="constant", fill_value=-1)
print("Constant imputed:\n", imp_const.fit_transform(X_missing))


# ══════════════════════════════════════════════════════════════════════
# 4. PIPELINE — Chain Everything Together
# ══════════════════════════════════════════════════════════════════════
print("\n── Pipeline ──")

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Simple pipeline: impute → scale → classify
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# fit() calls fit_transform on each step, then fit on the last
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print(f"Pipeline accuracy: {score:.3f}")

# Cross-validate the entire pipeline (prevents data leakage!)
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"CV scores: {cv_scores.round(3)}")
print(f"CV mean ± std: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Shorthand: make_pipeline (auto-generates step names)
pipe2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
pipe2.fit(X_train, y_train)


# ══════════════════════════════════════════════════════════════════════
# 5. COLUMNTRANSFORMER — Different Transforms for Different Columns
# ══════════════════════════════════════════════════════════════════════
print("\n── ColumnTransformer ──")

# Create a mixed-type dataset
df = pd.DataFrame({
    "age": [25, 30, np.nan, 45, 22],
    "income": [50000, 80000, 60000, np.nan, 35000],
    "city": ["NYC", "LA", "NYC", "Chicago", "LA"],
    "education": ["BS", "MS", "PhD", "BS", "MS"],
})
y_demo = [0, 1, 1, 0, 1]

print("Mixed dataset:")
print(df)

# Define separate pipelines for each column type
numeric_features = ["age", "income"]
categorical_features = ["city", "education"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combine with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Full pipeline: preprocess → model
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

full_pipeline.fit(df, y_demo)
print(f"\nPipeline fitted successfully")
print(f"Feature names: {preprocessor.get_feature_names_out()}")

# ── Auto-select columns by dtype ─────────────────────────────────────
preprocessor_auto = ColumnTransformer([
    ("num", numeric_transformer, make_column_selector(dtype_include="number")),
    ("cat", categorical_transformer, make_column_selector(dtype_include="object")),
])


# ══════════════════════════════════════════════════════════════════════
# 6. CUSTOM TRANSFORMERS
# ══════════════════════════════════════════════════════════════════════
print("\n── Custom transformers ──")

# FunctionTransformer wraps any function as a sklearn transformer
log_transformer = FunctionTransformer(np.log1p, validate=True)

income = np.array([[50000], [80000], [60000]])
print("Log-transformed income:", log_transformer.fit_transform(income).round(2).ravel())

# Use in a pipeline
pipe_custom = make_pipeline(
    FunctionTransformer(np.log1p),
    StandardScaler(),
)
print("Log + Scale:", pipe_custom.fit_transform(income).round(4).ravel())


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 1.1: Load the Titanic dataset. Build a ColumnTransformer that:
    - Numeric columns (Age, Fare): impute median → StandardScaler
    - Categorical columns (Sex, Embarked): impute mode → OneHotEncoder
    - Ordinal column (Pclass): leave as-is
    Create a full pipeline with LogisticRegression. Report CV accuracy.

Exercise 1.2: Compare the effect of StandardScaler vs MinMaxScaler vs
    no scaling on LogisticRegression and RandomForest accuracy.
    Which model is affected by scaling? Why?

Exercise 1.3: Create a custom transformer class (inheriting from
    BaseEstimator and TransformerMixin) that:
    - In fit(): learns the median of each column
    - In transform(): clips values to [median - 3*std, median + 3*std]
    Use it in a pipeline.

Exercise 1.4: Build a pipeline for a dataset with:
    - 5 numeric features (some with NaN)
    - 3 categorical features (some with NaN)
    - PolynomialFeatures(degree=2) on numeric features after scaling
    Cross-validate the full pipeline.
"""
