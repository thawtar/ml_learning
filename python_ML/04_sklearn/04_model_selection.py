"""
Module 4 — Lesson 4: Model Selection & Hyperparameter Tuning
=============================================================
Cross-validation strategies, grid search, randomized search,
and practical tips for choosing models in production ML.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    # Cross-validation
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, RepeatedStratifiedKFold,
    LeaveOneOut, TimeSeriesSplit,
    # Hyperparameter search
    GridSearchCV, RandomizedSearchCV,
    # Utilities
    train_test_split, learning_curve, validation_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer

# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════
X, y = make_classification(
    n_samples=800, n_features=20, n_informative=12,
    n_classes=2, random_state=42, flip_y=0.05,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)


# ══════════════════════════════════════════════════════════════════════
# 1. CROSS-VALIDATION STRATEGIES
# ══════════════════════════════════════════════════════════════════════
print("── CV Strategies ──")

rf = RandomForestClassifier(n_estimators=50, random_state=42)

# (a) Default k-fold (k=5)
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
print(f"5-Fold:              {scores.mean():.4f} ± {scores.std():.4f}")

# (b) Stratified k-fold (preserves class ratios — better for classification)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring="accuracy")
print(f"Stratified 5-Fold:   {scores.mean():.4f} ± {scores.std():.4f}")

# (c) Repeated stratified k-fold (more stable estimate)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(rf, X_train, y_train, cv=rskf, scoring="accuracy")
print(f"Repeated 5x3:        {scores.mean():.4f} ± {scores.std():.4f}")

# (d) Time-series split (NEVER shuffle time-series data!)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring="accuracy")
print(f"Time-Series Split:   {scores.mean():.4f} ± {scores.std():.4f}")

"""
WHICH CV TO USE?

  - Classification: StratifiedKFold (default for classifiers in sklearn)
  - Regression: KFold with shuffle=True
  - Time-series: TimeSeriesSplit (chronological order, no leakage!)
  - Small dataset: RepeatedKFold or LeaveOneOut
  - Large dataset: 5-fold or 10-fold is usually sufficient
"""


# ══════════════════════════════════════════════════════════════════════
# 2. cross_validate — RICHER THAN cross_val_score
# ══════════════════════════════════════════════════════════════════════
print("\n── cross_validate with multiple metrics ──")

# Returns multiple metrics + fit/score times
cv_results = cross_validate(
    rf, X_train, y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring=["accuracy", "f1", "roc_auc"],
    return_train_score=True,  # detect overfitting
    n_jobs=-1,
)

for key in ["test_accuracy", "test_f1", "test_roc_auc"]:
    vals = cv_results[key]
    print(f"  {key:>20s}: {vals.mean():.4f} ± {vals.std():.4f}")

# Check for overfitting: big gap between train and test score
for metric in ["accuracy", "f1", "roc_auc"]:
    train_mean = cv_results[f"train_{metric}"].mean()
    test_mean  = cv_results[f"test_{metric}"].mean()
    gap = train_mean - test_mean
    print(f"  {metric:>10s} gap (train-test): {gap:.4f}"
          f"  {'⚠ overfitting' if gap > 0.05 else '✓ OK'}")


# ══════════════════════════════════════════════════════════════════════
# 3. GRID SEARCH CV
# ══════════════════════════════════════════════════════════════════════
print("\n── GridSearchCV ──")

# Pipeline ensures scaling is inside CV (prevents data leakage!)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

# When using Pipeline, prefix param names with step name + "__"
param_grid = {
    "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["saga"],  # saga supports both l1 and l2
}

grid = GridSearchCV(
    pipe, param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="f1",
    refit=True,       # refit best model on full training data
    n_jobs=-1,
    verbose=0,
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV F1:  {grid.best_score_:.4f}")
print(f"Test F1:     {f1_score(y_test, grid.predict(X_test)):.4f}")

# Inspect results
results_df = pd.DataFrame(grid.cv_results_)[
    ["param_clf__C", "param_clf__penalty", "mean_test_score", "std_test_score", "rank_test_score"]
].sort_values("rank_test_score")
print(results_df.head(5).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
# 4. RANDOMIZED SEARCH CV
# ══════════════════════════════════════════════════════════════════════
print("\n── RandomizedSearchCV (much faster for large grids) ──")

from scipy.stats import uniform, randint, loguniform

# Random Forest has many hyperparameters → grid is huge
param_distributions = {
    "n_estimators":      randint(50, 300),
    "max_depth":         randint(3, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf":  randint(1, 10),
    "max_features":      uniform(0.1, 0.9),  # float → fraction of features
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=30,            # try 30 random combos (vs 1000s in grid)
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="f1",
    refit=True,
    random_state=42,
    n_jobs=-1,
)
random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best CV F1:  {random_search.best_score_:.4f}")
print(f"Test F1:     {f1_score(y_test, random_search.predict(X_test)):.4f}")

"""
GRID vs RANDOM SEARCH:

GridSearchCV:
  - Exhaustive: tries ALL combinations
  - Best when parameter space is small (< 100 combos)
  - Guaranteed to find the best combo in the grid

RandomizedSearchCV:
  - Samples n_iter combinations randomly
  - Scales much better to large/continuous parameter spaces
  - Often finds a "good enough" combo with far fewer iterations
  - Use scipy.stats distributions for continuous params (uniform, loguniform)

Rule of thumb:
  - ≤ 3 params, ≤ 5 values each → Grid
  - > 3 params or continuous ranges → Random
  - When you need even better: Optuna / Hyperopt (Bayesian optimization)
"""


# ══════════════════════════════════════════════════════════════════════
# 5. LEARNING CURVES — Diagnose Bias vs Variance
# ══════════════════════════════════════════════════════════════════════
print("\n── Learning curves ──")

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

print(f"{'Train Size':>12s} {'Train Score':>12s} {'Val Score':>12s} {'Gap':>8s}")
for size, tr, va in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(f"  {size:>10d}   {tr:.4f}       {va:.4f}     {tr-va:.4f}")

"""
READING LEARNING CURVES:

  High bias (underfitting):
    - Both train and val scores are LOW
    - They converge early
    → Fix: more complex model, more features, less regularization

  High variance (overfitting):
    - Train score high, val score low (big gap)
    - Gap shrinks slowly as data grows
    → Fix: more data, stronger regularization, simpler model, dropout
"""


# ══════════════════════════════════════════════════════════════════════
# 6. VALIDATION CURVES — Effect of One Hyperparameter
# ══════════════════════════════════════════════════════════════════════
print("\n── Validation curve (max_depth) ──")

param_range = np.arange(1, 20)

train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X_train, y_train,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

best_depth = param_range[np.argmax(val_scores.mean(axis=1))]
print(f"Best max_depth by validation score: {best_depth}")
print(f"{'max_depth':>10s} {'Train':>8s} {'Val':>8s}")
for d, tr, va in zip(param_range, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    marker = " ← best" if d == best_depth else ""
    print(f"  {d:>8d}   {tr:.4f}   {va:.4f}{marker}")


# ══════════════════════════════════════════════════════════════════════
# 7. CUSTOM SCORER
# ══════════════════════════════════════════════════════════════════════
print("\n── Custom scorer ──")

# Sometimes you need a metric sklearn doesn't provide
def specificity_score(y_true, y_pred):
    """True Negative Rate = TN / (TN + FP)"""
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Convert to a scorer object (higher is better ↔ greater_is_better=True)
specificity_scorer = make_scorer(specificity_score, greater_is_better=True)

scores = cross_val_score(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X_train, y_train, cv=5, scoring=specificity_scorer,
)
print(f"Specificity (custom): {scores.mean():.4f} ± {scores.std():.4f}")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 4.1: Compare StratifiedKFold(5), StratifiedKFold(10), and
    RepeatedStratifiedKFold(5, n_repeats=5) on a classifier of your
    choice. Which gives the most stable / reliable estimate?

Exercise 4.2: Perform GridSearchCV on an SVM (SVC) with:
    - C: [0.1, 1, 10, 100]
    - kernel: ["rbf", "poly"]
    - gamma: ["scale", "auto", 0.01, 0.1]
    Use a Pipeline with StandardScaler. Report the best params.

Exercise 4.3: Use RandomizedSearchCV with loguniform distributions
    for C and gamma on the SVM. Compare the best score to GridSearchCV.
    How many iterations does it need to match or beat the grid?

Exercise 4.4: Generate learning curves for both a DecisionTree (max_depth=3)
    and a RandomForest. Plot them and discuss which model has higher
    bias vs higher variance.

Exercise 4.5: Create a custom scorer for "balanced accuracy" from scratch
    (without using sklearn's balanced_accuracy_score). Verify it matches
    sklearn's implementation.
"""
