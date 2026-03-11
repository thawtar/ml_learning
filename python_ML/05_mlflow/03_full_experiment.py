"""
Module 5 — Lesson 3: Full Experiment Workflow
===============================================
End-to-end: load data → preprocess → train multiple models →
log everything → compare → select the best → register it.

This is the template you should follow in real ML projects.

Run the MLflow UI after this script:
    mlflow ui
    # Then open http://127.0.0.1:5000
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ── sklearn ──
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── mlflow ──
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
EXPERIMENT_NAME = "california_housing_regression"
MODEL_NAME = "housing_price_predictor"
RANDOM_STATE = 42
TEST_SIZE = 0.2

mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()


# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & EDA
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Data Loading & EDA")
print("=" * 60)

housing = fetch_california_housing(as_frame=True)
df = housing.frame  # DataFrame with features + target
feature_names = housing.feature_names
target_name = housing.target_names[0]

print(f"\nDataset shape: {df.shape}")
print(f"Features: {feature_names}")
print(f"Target: {target_name}")
print(f"\n{df.describe().round(3)}")

# Check for issues
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Target range: [{df[target_name].min():.2f}, {df[target_name].max():.2f}]")
print(f"Target mean:  {df[target_name].mean():.2f}")


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 2: Feature Engineering")
print("=" * 60)

X = df[feature_names].copy()
y = df[target_name].values

# Create meaningful interaction features
X["rooms_per_household"] = X["AveRooms"] / X["AveOccup"].clip(lower=0.1)
X["bedrooms_ratio"] = X["AveBedrms"] / X["AveRooms"].clip(lower=0.1)
X["population_per_household"] = X["Population"] / X["HouseAge"].clip(lower=0.1)

all_features = list(X.columns)
print(f"Features after engineering: {len(all_features)}")
print(f"New features: rooms_per_household, bedrooms_ratio, population_per_household")

# Split BEFORE any preprocessing (prevent leakage!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ══════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 3: Build Preprocessing Pipeline")
print("=" * 60)

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"Preprocessed shapes: train={X_train_processed.shape}, test={X_test_processed.shape}")


# ══════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING & COMPARISON
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 4: Train & Compare Models")
print("=" * 60)

# Define model candidates
model_configs = [
    {
        "name": "Ridge (α=1.0)",
        "model": Ridge(alpha=1.0),
        "params": {"alpha": 1.0, "model_type": "Ridge"},
    },
    {
        "name": "Ridge (α=10.0)",
        "model": Ridge(alpha=10.0),
        "params": {"alpha": 10.0, "model_type": "Ridge"},
    },
    {
        "name": "Lasso (α=0.1)",
        "model": Lasso(alpha=0.1),
        "params": {"alpha": 0.1, "model_type": "Lasso"},
    },
    {
        "name": "ElasticNet",
        "model": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "params": {"alpha": 0.1, "l1_ratio": 0.5, "model_type": "ElasticNet"},
    },
    {
        "name": "Random Forest",
        "model": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "params": {"n_estimators": 100, "max_depth": 10, "model_type": "RandomForest"},
    },
    {
        "name": "Gradient Boosting",
        "model": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
        ),
        "params": {
            "n_estimators": 200, "max_depth": 5,
            "learning_rate": 0.1, "model_type": "GradientBoosting",
        },
    },
]


def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    """Train, predict, compute metrics."""
    model.fit(X_tr, y_tr)
    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)

    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(y_tr, y_pred_train))),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_te, y_pred_test))),
        "test_mae":   float(mean_absolute_error(y_te, y_pred_test)),
        "test_r2":    float(r2_score(y_te, y_pred_test)),
        "overfit_gap": float(
            np.sqrt(mean_squared_error(y_te, y_pred_test))
            - np.sqrt(mean_squared_error(y_tr, y_pred_train))
        ),
    }
    return model, metrics, y_pred_test


# Train all models and log to MLflow
all_results = []

for config in model_configs:
    with mlflow.start_run(run_name=config["name"]):
        # Log config
        mlflow.log_params(config["params"])
        mlflow.log_params({
            "n_features": X_train_processed.shape[1],
            "n_train_samples": X_train_processed.shape[0],
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        })
        mlflow.set_tag("stage", "experiment")

        # Train & evaluate
        trained_model, metrics, y_pred = evaluate_model(
            config["model"],
            X_train_processed, X_test_processed,
            y_train, y_test,
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model with signature
        signature = infer_signature(X_train_processed, y_pred)
        mlflow.sklearn.log_model(
            trained_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train_processed[:3],
        )

        # Log residual stats as artifact
        residuals = y_test - y_pred
        residual_stats = {
            "mean": float(residuals.mean()),
            "std": float(residuals.std()),
            "max_abs": float(np.abs(residuals).max()),
            "percentile_95": float(np.percentile(np.abs(residuals), 95)),
        }
        mlflow.log_dict(residual_stats, "residual_stats.json")

        all_results.append({
            "name": config["name"],
            "run_id": mlflow.active_run().info.run_id,
            **metrics,
        })

        status = "⚠ overfit" if metrics["overfit_gap"] > 0.1 else "✓"
        print(f"  {config['name']:>25s}  RMSE={metrics['test_rmse']:.4f}  "
              f"R²={metrics['test_r2']:.4f}  {status}")


# ══════════════════════════════════════════════════════════════════════
# 5. SELECT & REGISTER THE BEST MODEL
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 5: Select & Register Best Model")
print("=" * 60)

results_df = pd.DataFrame(all_results).sort_values("test_rmse")
print("\n── Leaderboard ──")
print(results_df[["name", "test_rmse", "test_mae", "test_r2", "overfit_gap"]].to_string(index=False))

best = results_df.iloc[0]
print(f"\n★ Best model: {best['name']} (RMSE={best['test_rmse']:.4f}, R²={best['test_r2']:.4f})")

# Register the best model
best_run_id = best["run_id"]
model_uri = f"runs:/{best_run_id}/model"

try:
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version
    print(f"Registered as {MODEL_NAME} version {version}")

    # Set alias
    client.set_registered_model_alias(MODEL_NAME, "champion", str(version))
    print(f"Set 'champion' alias on version {version}")

    # Add description
    client.update_registered_model(
        name=MODEL_NAME,
        description=f"California Housing price predictor. "
                    f"Best model: {best['name']} with RMSE={best['test_rmse']:.4f}",
    )
except Exception as e:
    print(f"Registration note: {e}")


# ══════════════════════════════════════════════════════════════════════
# 6. LOAD & VALIDATE THE CHAMPION MODEL
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 6: Load & Validate Champion")
print("=" * 60)

try:
    champion_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    val_preds = champion_model.predict(X_test_processed)
    val_rmse = np.sqrt(mean_squared_error(y_test, val_preds))
    print(f"Champion model RMSE on test set: {val_rmse:.4f}")
    print(f"Sample predictions vs actuals:")
    for i in range(5):
        print(f"  Predicted: {val_preds[i]:.3f}  Actual: {y_test[i]:.3f}  "
              f"Error: {abs(val_preds[i] - y_test[i]):.3f}")
except Exception as e:
    # Fallback: load by run ID
    champion_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    val_preds = champion_model.predict(X_test_processed)
    val_rmse = np.sqrt(mean_squared_error(y_test, val_preds))
    print(f"Loaded by run_id — RMSE: {val_rmse:.4f}")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("EXPERIMENT COMPLETE")
print("=" * 60)
print(f"""
What we did:
  1. Loaded and explored the California Housing dataset
  2. Engineered 3 new features (room ratios, population density)
  3. Built a preprocessing pipeline (impute → scale)
  4. Trained 6 models, logging everything to MLflow
  5. Selected the best by RMSE and registered it
  6. Loaded the champion model and validated it

Next steps:
  - Run `mlflow ui` to explore the runs visually
  - Try adding more models (XGBoost, LightGBM)
  - Tune hyperparameters with RandomizedSearchCV + MLflow
  - Add cross-validation scores to the comparison
  - Deploy the champion model with `mlflow models serve`
""")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 3.1: Add cross-validation (5-fold) to the comparison. Log
    both CV RMSE and holdout RMSE. Do they agree?

Exercise 3.2: Add hyperparameter tuning for the Gradient Boosting
    model using RandomizedSearchCV. Log the best params to MLflow.
    Does the tuned model beat the default?

Exercise 3.3: Create a "model card" artifact for the champion model
    containing: model type, training date, dataset info, performance
    metrics, known limitations. Log it as a JSON artifact.

Exercise 3.4: Write a script that loads the champion model and serves
    predictions via a simple function. Test it with synthetic data.

Exercise 3.5: (Advanced) Build a nightly retraining simulation:
    - Split data into 5 "time periods"
    - For each period, train on all prior data
    - Log each as a new run
    - Promote the best version to champion
    - Track how performance evolves as more data arrives
"""
