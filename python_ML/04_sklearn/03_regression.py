"""
Module 4 — Lesson 3: Regression
================================
Regression models, regularization, and evaluation metrics.
Covers linear models, tree ensembles, and proper regression diagnostics.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_regression, fetch_california_housing,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ── Regressors ──
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.svm import SVR

# ── Metrics ──
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error,
)


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════
X, y = make_regression(
    n_samples=1000, n_features=20, n_informative=12,
    noise=10.0, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42,
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")


# ══════════════════════════════════════════════════════════════════════
# 1. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n── Model comparison ──")

models = {
    "Linear Regression":    make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge (α=1.0)":        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso (α=1.0)":        make_pipeline(StandardScaler(), Lasso(alpha=1.0)),
    "ElasticNet":           make_pipeline(StandardScaler(), ElasticNet(alpha=1.0, l1_ratio=0.5)),
    "Decision Tree":        DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR (RBF)":            make_pipeline(StandardScaler(), SVR()),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "RMSE":  np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE":   mean_absolute_error(y_test, y_pred),
        "R²":    r2_score(y_test, y_pred),
        "MAPE":  mean_absolute_percentage_error(y_test, y_pred),
    })

results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
# 2. REGULARIZATION DEEP DIVE
# ══════════════════════════════════════════════════════════════════════
"""
REGULARIZATION CHEAT SHEET:

     │ Penalty         │ Effect on coefficients
─────┼─────────────────┼────────────────────────────────────
Ridge│ L2: α·Σ(βᵢ²)   │ Shrinks all coefficients (small)
Lasso│ L1: α·Σ|βᵢ|    │ Drives some coefficients to ZERO (feature selection!)
E-Net│ α(ρ·L1 + ½(1-ρ)·L2) │ Combo of both

α (alpha): regularization strength.   ↑ = more shrinkage
l1_ratio (ρ): 1 = pure Lasso, 0 = pure Ridge
"""

print("\n── Regularization: Lasso for feature selection ──")

# Fit Lasso and inspect which features survive
lasso = make_pipeline(StandardScaler(), Lasso(alpha=1.0))
lasso.fit(X_train, y_train)

coefs = lasso.named_steps["lasso"].coef_
n_nonzero = np.sum(coefs != 0)
print(f"Features with non-zero coefficient: {n_nonzero} / {X_train.shape[1]}")
print(f"Zero coefficients: {np.sum(coefs == 0)}")

# Ridge alpha sweep
print("\n── Ridge regularization path ──")
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    ridge.fit(X_train, y_train)
    score = ridge.score(X_test, y_test)
    coef_norm = np.linalg.norm(ridge.named_steps["ridge"].coef_)
    print(f"  α={alpha:>6.2f}  R²={score:.4f}  ||coef||={coef_norm:.2f}")


# ══════════════════════════════════════════════════════════════════════
# 3. POLYNOMIAL REGRESSION (Underfitting → Overfitting)
# ══════════════════════════════════════════════════════════════════════
print("\n── Polynomial regression: bias-variance tradeoff ──")

# Simple 1D data with sine wave
rng = np.random.default_rng(42)
X_1d = np.sort(rng.uniform(0, 6, size=(100, 1)), axis=0)
y_1d = np.sin(X_1d.ravel()) + rng.normal(0, 0.2, size=100)
X_1d_tr, X_1d_te, y_1d_tr, y_1d_te = train_test_split(
    X_1d, y_1d, test_size=0.3, random_state=42,
)

for degree in [1, 3, 5, 10, 20]:
    pipe = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )
    pipe.fit(X_1d_tr, y_1d_tr)
    train_r2 = pipe.score(X_1d_tr, y_1d_tr)
    test_r2  = pipe.score(X_1d_te, y_1d_te)
    print(f"  Degree {degree:>2d}:  Train R²={train_r2:.4f}  Test R²={test_r2:.4f}"
          f"  {'← underfitting' if test_r2 < 0.5 else '← overfitting' if train_r2 - test_r2 > 0.3 else '← good'}")


# ══════════════════════════════════════════════════════════════════════
# 4. REGRESSION METRICS EXPLAINED
# ══════════════════════════════════════════════════════════════════════
"""
METRIC CHEAT SHEET:

Metric    │ Formula                    │ Notes
──────────┼────────────────────────────┼──────────────────────────────
MSE       │ mean((y - ŷ)²)            │ Penalizes large errors (squared)
RMSE      │ √MSE                      │ Same units as y
MAE       │ mean(|y - ŷ|)             │ Robust to outliers (linear penalty)
R²        │ 1 - SS_res / SS_tot       │ 1 = perfect, 0 = mean baseline, <0 = worse than mean
Adj. R²   │ 1-(1-R²)(n-1)/(n-p-1)    │ Penalizes adding useless features
MAPE      │ mean(|y - ŷ| / |y|)       │ % error. ⚠ undefined when y=0

ML convention:
  - Sklearn's scoring uses "neg_mean_squared_error" (higher is better)
  - So scores are NEGATIVE MSE/MAE when using cross_val_score
"""

# Demonstration of scoring conventions
print("\n── Scoring convention gotcha ──")
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
neg_mse = cross_val_score(ridge, X, y, cv=5, scoring="neg_mean_squared_error")
print(f"neg_MSE scores: {neg_mse.round(2)}")
print(f"Actual MSE: {(-neg_mse).round(2)}")
print(f"RMSE: {np.sqrt(-neg_mse).round(2)}")


# ══════════════════════════════════════════════════════════════════════
# 5. RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n── Residual analysis (Gradient Boosting) ──")

gb = models["Gradient Boosting"]
y_pred_gb = gb.predict(X_test)

residuals = y_test - y_pred_gb
print(f"Residual mean:   {residuals.mean():.4f}  (should be ≈ 0)")
print(f"Residual std:    {residuals.std():.4f}")
print(f"Max |residual|:  {np.abs(residuals).max():.4f}")
print(f"Skewness:        {pd.Series(residuals).skew():.4f}  (should be ≈ 0)")

# Check if residuals are roughly normally distributed
from scipy.stats import shapiro
_, p_value = shapiro(residuals[:50])  # Shapiro-Wilk has a sample limit
print(f"Shapiro-Wilk p-value (n=50): {p_value:.4f}  {'✓ normal' if p_value > 0.05 else '✗ non-normal'}")


# ══════════════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE (Tree-Based)
# ══════════════════════════════════════════════════════════════════════
print("\n── Feature importance (Gradient Boosting) ──")

importances = gb.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("Top 10 features:")
for rank, idx in enumerate(sorted_idx[:10], 1):
    print(f"  {rank:>2d}. Feature {idx:>2d} → importance {importances[idx]:.4f}")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 3.1: Load California Housing dataset. Train Ridge with 5
    different alpha values. Plot Train R² vs Test R² as a function of alpha.

Exercise 3.2: Use Lasso with increasing alpha to find the number of
    selected features at each alpha level. What alpha keeps only the
    top 5 most important features?

Exercise 3.3: Generate a noisy sine wave. Fit polynomials of degree
    1 through 15. Plot train and test RMSE vs degree. Identify the
    sweet spot (bias-variance tradeoff).

Exercise 3.4: Train a GradientBoostingRegressor. Compute and plot:
    (a) residuals vs predicted values (check for patterns)
    (b) histogram of residuals (check for normality)
    (c) Q-Q plot

Exercise 3.5: Compare predictions of Ridge, Lasso and ElasticNet
    on California Housing. For each, report the 5 largest absolute
    coefficients and discuss what they mean.
"""
