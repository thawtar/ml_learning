"""
Module 5 — Lesson 1: MLflow Tracking Basics
=============================================
Log experiments reproducibly: parameters, metrics, artifacts, and models.
MLflow is the de facto open-source experiment tracking standard.

Setup:
    pip install mlflow

Quick start (run MLflow UI after this script):
    mlflow ui          # opens http://127.0.0.1:5000
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ── sklearn ──
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ── mlflow ──
import mlflow
import mlflow.sklearn


# ══════════════════════════════════════════════════════════════════════
# SETUP — Use a local tracking directory (no server needed)
# ══════════════════════════════════════════════════════════════════════
# By default, MLflow writes to ./mlruns in the current dir.
# You can change it:
# mlflow.set_tracking_uri("sqlite:///mlflow.db")   # local SQLite
# mlflow.set_tracking_uri("http://localhost:5000")  # remote server

print(f"Tracking URI: {mlflow.get_tracking_uri()}")


# ══════════════════════════════════════════════════════════════════════
# 1. YOUR FIRST RUN — Manual Logging
# ══════════════════════════════════════════════════════════════════════
print("\n── Manual logging ──")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target,
)

# Set an experiment name (creates it if it doesn't exist)
mlflow.set_experiment("iris_classification")

# Start a run — everything inside is tracked
with mlflow.start_run(run_name="rf_manual_v1"):
    # ── Hyperparameters ──
    n_estimators = 100
    max_depth = 5

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.3)

    # ── Train ──
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ── Metrics ──
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_weighted", f1)
    print(f"Manual run — Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # ── Log multiple params/metrics at once ──
    mlflow.log_params({"scaler": "none", "cv_folds": 5})
    mlflow.log_metrics({"precision": 0.95, "recall": 0.94})  # just examples

    # ── Tags (searchable metadata) ──
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("author", "ml_learner")

    # ── Artifacts (any file) ──
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)
    os.remove(report_path)  # clean up local file

    # ── Log the model itself ──
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Run ID: {mlflow.active_run().info.run_id}")


# ══════════════════════════════════════════════════════════════════════
# 2. LOG PARAMS IN BULK — log_params & log_metrics
# ══════════════════════════════════════════════════════════════════════
print("\n── Bulk logging ──")

with mlflow.start_run(run_name="rf_bulk_params"):
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "criterion": "gini",
    }
    mlflow.log_params(params)

    model = RandomForestClassifier(**{k: v for k, v in params.items()
                                      if k != "criterion"}, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }
    mlflow.log_metrics(metrics)
    print(f"Bulk run — {metrics}")


# ══════════════════════════════════════════════════════════════════════
# 3. STEP METRICS — Track Over Training Epochs
# ══════════════════════════════════════════════════════════════════════
print("\n── Step metrics (simulating training epochs) ──")

with mlflow.start_run(run_name="epoch_tracking"):
    mlflow.log_param("model", "simulated_nn")

    # Simulate training loop
    for epoch in range(1, 11):
        train_loss = 1.0 / epoch + np.random.normal(0, 0.02)
        val_loss   = 1.0 / epoch + 0.1 + np.random.normal(0, 0.03)
        train_acc  = 1 - train_loss + np.random.normal(0, 0.01)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("train_acc", max(0, train_acc), step=epoch)

    print("Logged 10 epochs of train/val loss + accuracy")


# ══════════════════════════════════════════════════════════════════════
# 4. AUTOLOG — Zero-Effort Logging
# ══════════════════════════════════════════════════════════════════════
print("\n── Autolog ──")

# mlflow.sklearn.autolog() automatically logs:
#   - Parameters (all constructor args)
#   - Metrics (accuracy for classifiers, RMSE for regressors)
#   - The fitted model as an artifact
#   - Feature importances (for tree models)
#   - Training dataset info

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="rf_autolog"):
    model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
    model.fit(X_train, y_train)

    # Everything is logged automatically!
    score = model.score(X_test, y_test)
    print(f"Autolog run — Test accuracy: {score:.4f}")

# Disable autolog when you want manual control again
mlflow.sklearn.autolog(disable=True)


# ══════════════════════════════════════════════════════════════════════
# 5. LOGGING ARTIFACTS — Figures, Data, Configs
# ══════════════════════════════════════════════════════════════════════
print("\n── Artifacts ──")

with mlflow.start_run(run_name="artifact_demo"):
    # Log a JSON config
    config = {"learning_rate": 0.01, "batch_size": 32, "epochs": 50}
    config_path = "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    mlflow.log_artifact(config_path)
    os.remove(config_path)

    # Log a dict directly as JSON
    mlflow.log_dict(config, "config_v2.json")

    # Log a numpy array
    importances = np.random.rand(10)
    np.save("feature_importances.npy", importances)
    mlflow.log_artifact("feature_importances.npy")
    os.remove("feature_importances.npy")

    # Log a pandas DataFrame as CSV
    df = pd.DataFrame({"feature": [f"f{i}" for i in range(10)], "importance": importances})
    csv_path = "feature_importance.csv"
    df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)
    os.remove(csv_path)

    print("Logged JSON, numpy, and CSV artifacts")


# ══════════════════════════════════════════════════════════════════════
# 6. QUERYING PAST RUNS
# ══════════════════════════════════════════════════════════════════════
print("\n── Querying runs ──")

experiment = mlflow.get_experiment_by_name("iris_classification")
if experiment:
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
    )
    print(f"Total runs: {len(runs)}")
    cols = ["run_id", "metrics.accuracy", "metrics.f1_weighted", "params.n_estimators"]
    available_cols = [c for c in cols if c in runs.columns]
    if available_cols:
        print(runs[available_cols].head().to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 1.1: Train 3 different classifiers (Logistic Regression,
    Random Forest, SVM) on the Iris dataset. Log each as a separate
    run in the same experiment. Include all hyperparams and metrics.

Exercise 1.2: Create a training loop that simulates 20 epochs.
    Log train_loss, val_loss, and val_accuracy at each step.
    Open the MLflow UI and inspect the step metric charts.

Exercise 1.3: Use mlflow.sklearn.autolog() to train a GradientBoosting
    classifier. Inspect the logged artifacts. What did autolog capture
    that you wouldn't have logged manually?

Exercise 1.4: Write a function that accepts a model and dataset,
    trains it, and logs everything to MLflow (params, metrics, model,
    classification report as artifact). Test it with 3 different models.

Exercise 1.5: Use mlflow.search_runs() to find the run with the
    highest accuracy in your experiment. Load that model and make
    predictions on new data.
"""
