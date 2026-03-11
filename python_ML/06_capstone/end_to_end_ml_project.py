"""
Module 6 — Capstone: End-to-End ML Project
============================================
Bring everything together: EDA, cleaning, feature engineering,
model selection, hyperparameter tuning, evaluation, and experiment
tracking with MLflow.

Dataset: California Housing (regression) — predict median house value.

Instructions:
  This file provides the SKELETON. Fill in every TODO block.
  Each TODO references a skill from a previous module.
  Run with: python end_to_end_ml_project.py
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── sklearn ──
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform

# ── mlflow ──
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ══════════════════════════════════════════════════════════════════════
# PART 1: DATA LOADING & EDA  (Module 2 — Pandas)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 1: Data Loading & Exploration")
print("=" * 60)

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
feature_names = housing.feature_names
target = "MedHouseVal"

# TODO 1.1: Print the shape, dtypes, and first 5 rows
# Hint: df.shape, df.dtypes, df.head()


# TODO 1.2: Use .describe() to get summary statistics
# Print the result


# TODO 1.3: Check for missing values and duplicates
# Hint: df.isnull().sum(), df.duplicated().sum()


# TODO 1.4: Print the correlation between each feature and the target
# Hint: df.corr()[target].sort_values()


print("Part 1 complete.\n")


# ══════════════════════════════════════════════════════════════════════
# PART 2: DATA CLEANING  (Module 2 — Pandas)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 2: Data Cleaning")
print("=" * 60)

# TODO 2.1: Check for and remove any duplicate rows
# Hint: df.drop_duplicates(inplace=True)


# TODO 2.2: Detect outliers in the target column using IQR method
# Remove rows where target > Q3 + 1.5*IQR or target < Q1 - 1.5*IQR
# Hint: Q1 = df[target].quantile(0.25), Q3 = df[target].quantile(0.75)


# TODO 2.3: Print the cleaned dataset shape
# print(f"Shape after cleaning: {df.shape}")


print("Part 2 complete.\n")


# ══════════════════════════════════════════════════════════════════════
# PART 3: FEATURE ENGINEERING  (Module 2 — Pandas, Module 1 — NumPy)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 3: Feature Engineering")
print("=" * 60)

# TODO 3.1: Create "rooms_per_household" = AveRooms / AveOccup
# Clip AveOccup to avoid division by tiny numbers


# TODO 3.2: Create "bedrooms_ratio" = AveBedrms / AveRooms
# Clip AveRooms similarly


# TODO 3.3: Create "people_per_household" = Population / HouseAge
# Clip HouseAge


# TODO 3.4: Bin "HouseAge" into categories: 'new' (0-10), 'mid' (10-30), 'old' (30+)
# Use pd.cut() → then one-hot encode with pd.get_dummies()


# TODO 3.5: Print all feature names after engineering
# print(f"Features after engineering: {list(df.columns)}")


print("Part 3 complete.\n")


# ══════════════════════════════════════════════════════════════════════
# PART 4: TRAIN/TEST SPLIT & PREPROCESSING PIPELINE
# (Module 4 — Sklearn)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 4: Split & Preprocessing")
print("=" * 60)

# Separate features and target
X = df.drop(columns=[target])
y = df[target].values

# TODO 4.1: Split into train/test (80/20, stratify is not needed for regression)
# X_train, X_test, y_train, y_test = ...


# TODO 4.2: Build a preprocessing Pipeline with:
#   - SimpleImputer(strategy="median")
#   - StandardScaler()
# preprocessor = Pipeline([...])


# TODO 4.3: Fit on train, transform both train and test
# X_train_proc = preprocessor.fit_transform(X_train)
# X_test_proc  = preprocessor.transform(X_test)


print("Part 4 complete.\n")


# ══════════════════════════════════════════════════════════════════════
# PART 5: MODEL TRAINING & COMPARISON  (Module 4 — Sklearn)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 5: Model Training & Comparison")
print("=" * 60)

# TODO 5.1: Define at least 3 models to compare:
#   - Ridge(alpha=1.0)
#   - RandomForestRegressor(n_estimators=100, max_depth=10)
#   - GradientBoostingRegressor(n_estimators=200, learning_rate=0.1)

models = {
    # "Ridge": Ridge(alpha=1.0),
    # "Random Forest": RandomForestRegressor(...),
    # "Gradient Boosting": GradientBoostingRegressor(...),
}

# TODO 5.2: Train each model, compute RMSE, MAE, R² on test set
# Print a comparison table
# for name, model in models.items():
#     model.fit(X_train_proc, y_train)
#     y_pred = model.predict(X_test_proc)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     ...


# TODO 5.3: Run 5-fold cross-validation on each model
# Use cross_val_score with scoring="neg_root_mean_squared_error"
# Report mean ± std


print("Part 5 complete.\n")


# ══════════════════════════════════════════════════════════════════════
# PART 6: HYPERPARAMETER TUNING  (Module 4 — Sklearn)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 6: Hyperparameter Tuning")
print("=" * 60)

# TODO 6.1: Use RandomizedSearchCV to tune GradientBoostingRegressor
# param_distributions = {
#     "n_estimators": randint(100, 500),
#     "max_depth": randint(3, 15),
#     "learning_rate": uniform(0.01, 0.3),
#     "min_samples_split": randint(2, 20),
#     "min_samples_leaf": randint(1, 10),
# }
# search = RandomizedSearchCV(
#     GradientBoostingRegressor(random_state=42),
#     param_distributions,
#     n_iter=20, cv=5, scoring="neg_root_mean_squared_error",
#     random_state=42, n_jobs=-1,
# )
# search.fit(X_train_proc, y_train)


# TODO 6.2: Print the best params and score
# print(f"Best params: {search.best_params_}")
# print(f"Best CV RMSE: {-search.best_score_:.4f}")


# TODO 6.3: Evaluate the tuned model on the test set
# best_model = search.best_estimator_
# y_pred = best_model.predict(X_test_proc)
# print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")


print("Part 6 complete.\n")


# ══════════════════════════════════════════════════════════════════════
# PART 7: EXPERIMENT TRACKING WITH MLFLOW  (Module 5 — MLflow)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 7: MLflow Experiment Tracking")
print("=" * 60)

mlflow.set_experiment("capstone_housing")

# TODO 7.1: Log each model from Part 5 as a separate MLflow run
# For each model:
#   - mlflow.log_params({...})
#   - mlflow.log_metrics({"rmse": ..., "mae": ..., "r2": ...})
#   - mlflow.sklearn.log_model(model, "model", signature=...)
#   - mlflow.set_tag("model_type", name)

# for name, model in models.items():
#     with mlflow.start_run(run_name=name):
#         ...


# TODO 7.2: Log the tuned model from Part 6 as a separate run
#   Include all best hyperparameters and test metrics
#   Add tag: "tuned": "yes"


# TODO 7.3: Use mlflow.search_runs() to find the best run by RMSE
# experiment = mlflow.get_experiment_by_name("capstone_housing")
# runs = mlflow.search_runs(
#     experiment_ids=[experiment.experiment_id],
#     order_by=["metrics.rmse ASC"],
# )
# print(runs[["run_id", "metrics.rmse", "metrics.r2"]].head())


# TODO 7.4: Register the best model
# best_run_id = runs.iloc[0]["run_id"]
# mlflow.register_model(f"runs:/{best_run_id}/model", "housing_capstone")


print("Part 7 complete.\n")


# ══════════════════════════════════════════════════════════════════════
# PART 8: FINAL REPORT  (Bringing it all together)
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 8: Final Report")
print("=" * 60)

# TODO 8.1: Print a final summary including:
#   - Dataset info (shape, features)
#   - Best model name and its hyperparameters
#   - Test metrics (RMSE, MAE, R²)
#   - Number of MLflow runs logged


# TODO 8.2: Print residual analysis of the best model
#   - Mean residual (should be ≈ 0)
#   - Std of residuals
#   - 95th percentile of |residual|


# TODO 8.3: Print feature importance (if tree-based model)
#   Sort by importance, print top 5


print("\n🎉 Capstone complete! Run `mlflow ui` to explore your experiments.")


# ══════════════════════════════════════════════════════════════════════
# BONUS CHALLENGES
# ══════════════════════════════════════════════════════════════════════
"""
Bonus 1: Add visualization — plot actual vs predicted, residual
    distribution, and feature importances. Save plots as MLflow artifacts.

Bonus 2: Try XGBoost or LightGBM (pip install xgboost lightgbm).
    Compare with sklearn models. Log to the same MLflow experiment.

Bonus 3: Implement a simple model serving function:
    def predict_house_price(features: dict) -> float:
        Load the champion model from MLflow, preprocess the
        input, and return the prediction.

Bonus 4: Add a data validation step: check that input features
    are within expected ranges before making predictions.

Bonus 5: Create a Jupyter notebook version of this capstone with
    inline plots and markdown explanations.
"""
