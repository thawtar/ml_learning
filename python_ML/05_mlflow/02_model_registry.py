"""
Module 5 — Lesson 2: Model Registry & Model Management
========================================================
Versioning, staging, and serving models with MLflow.
The Model Registry is how teams move from "notebook experiment"
to "reproducible, deployable model."

Prerequisite: Run 01_tracking_basics.py first to have some logged runs.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target,
)

mlflow.set_experiment("iris_model_registry")
client = MlflowClient()

MODEL_NAME = "iris_classifier"


# ══════════════════════════════════════════════════════════════════════
# 1. REGISTER A MODEL DURING TRAINING
# ══════════════════════════════════════════════════════════════════════
print("── Register model during training ──")

with mlflow.start_run(run_name="rf_registry_v1") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_params({"n_estimators": 100, "max_depth": 5})
    mlflow.log_metric("accuracy", acc)

    # This BOTH logs the model artifact AND registers it in the Model Registry
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,  # ← triggers registration
    )
    run_id_v1 = run.info.run_id
    print(f"Registered {MODEL_NAME} v1 — accuracy: {acc:.4f}")


# ══════════════════════════════════════════════════════════════════════
# 2. REGISTER MORE VERSIONS
# ══════════════════════════════════════════════════════════════════════
print("\n── Register additional versions ──")

configs = [
    ("GradientBoosting v2", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ("Pipeline LR v3", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
]

for name, model in configs:
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        f1 = f1_score(y_test, model.predict(X_test), average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        print(f"  {name}: accuracy={acc:.4f}, f1={f1:.4f}")


# ══════════════════════════════════════════════════════════════════════
# 3. INSPECT THE MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════
print("\n── Model Registry contents ──")

try:
    # List all versions of our model
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    print(f"Model '{MODEL_NAME}' has {len(versions)} version(s):")
    for v in versions:
        print(f"  Version {v.version}: run_id={v.run_id[:8]}... "
              f"status={v.status}")
except Exception as e:
    print(f"  (Registry query failed: {e})")


# ══════════════════════════════════════════════════════════════════════
# 4. MODEL ALIASES (MLflow 2.x)
# ══════════════════════════════════════════════════════════════════════
print("\n── Model aliases ──")

"""
MLflow 2.x replaced Stages (Staging/Production/Archived) with Aliases.
Aliases are flexible labels you can attach to any model version.

Common aliases:
  - "champion"    → the model currently serving in production
  - "challenger"  → a candidate model being A/B tested
  - "latest"      → most recently trained version
"""

try:
    # Set alias on version 1
    client.set_registered_model_alias(MODEL_NAME, "champion", "1")
    print(f"Set 'champion' alias on version 1")

    # Get model by alias
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    print(f"Champion: version {champion.version}, run_id={champion.run_id[:8]}...")

except Exception as e:
    print(f"  (Alias operations require MLflow 2.x: {e})")


# ══════════════════════════════════════════════════════════════════════
# 5. LOAD A REGISTERED MODEL
# ══════════════════════════════════════════════════════════════════════
print("\n── Load registered models ──")

# Method 1: Load by version number
try:
    model_v1 = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")
    preds = model_v1.predict(X_test[:5])
    print(f"Model v1 predictions: {preds}")
except Exception as e:
    print(f"  Load by version failed: {e}")

# Method 2: Load by alias (MLflow 2.x)
try:
    model_champ = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    preds = model_champ.predict(X_test[:5])
    print(f"Champion predictions:  {preds}")
except Exception as e:
    print(f"  Load by alias failed: {e}")

# Method 3: Load by run_id (always works)
try:
    model_run = mlflow.sklearn.load_model(f"runs:/{run_id_v1}/model")
    preds = model_run.predict(X_test[:5])
    print(f"Run-based predictions: {preds}")
except Exception as e:
    print(f"  Load by run_id failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# 6. MODEL COMPARISON & PROMOTION WORKFLOW
# ══════════════════════════════════════════════════════════════════════
print("\n── Compare versions and promote ──")

try:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    best_version = None
    best_acc = -1

    for v in versions:
        run = client.get_run(v.run_id)
        acc = run.data.metrics.get("accuracy", 0)
        print(f"  Version {v.version}: accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_version = v.version

    if best_version:
        client.set_registered_model_alias(MODEL_NAME, "champion", str(best_version))
        print(f"\n  → Promoted version {best_version} to 'champion' (acc={best_acc:.4f})")

except Exception as e:
    print(f"  Comparison failed: {e}")

"""
TYPICAL PROMOTION WORKFLOW:

    1. Train multiple models → log to MLflow
    2. Register best candidate → new version in Model Registry
    3. Set alias "challenger" on the new version
    4. Run A/B test or shadow mode comparison
    5. If challenger wins → move "champion" alias to it
    6. Old champion gets "archived" alias or is deleted

    In production (with MLflow + CI/CD):
      - GitHub Actions / Jenkins trigger training
      - Model is registered automatically
      - Automated tests validate the model
      - Promotion happens via API (not manually!)
"""


# ══════════════════════════════════════════════════════════════════════
# 7. MODEL SIGNATURE & INPUT EXAMPLE
# ══════════════════════════════════════════════════════════════════════
print("\n── Model signature ──")

from mlflow.models.signature import infer_signature

with mlflow.start_run(run_name="with_signature"):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Infer schema from data
    signature = infer_signature(X_train, y_pred)
    print(f"Signature: {signature}")

    # Log with signature + input example (enables model serving validation)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_train[:3],  # saved for documentation + serving tests
    )
    print("Model logged with signature and input example")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 2.1: Train 3 different classifiers. Register all of them under
    the same model name. Then write code to find the best version by F1
    score and set it as "champion".

Exercise 2.2: Load the champion model by alias. Make predictions on
    new data. Verify it gives the same predictions as the original model.

Exercise 2.3: Add a model description and tags to a registered model
    using the MlflowClient. Verify they appear in the registry.

Exercise 2.4: Write a function that:
    (a) Trains a model
    (b) Logs it with a signature and input example
    (c) Registers it
    (d) Compares it to the current champion
    (e) Promotes it if it's better
    This simulates a basic CI/CD pipeline for ML.

Exercise 2.5: Research: What is the MLflow Model format? What files
    are in the artifact directory? (Hint: MLmodel, conda.yaml,
    python_env.yaml, requirements.txt, model.pkl)
"""
