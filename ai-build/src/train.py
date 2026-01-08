"""Training script with MLflow experiment tracking."""
import os
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

DATA_DIR = "../data_processed"
MODEL_DIR = "../models"
PARAMS_FILE = "../params.yaml"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed training and test data."""
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def load_params() -> dict:
    """Load hyperparameters from params.yaml."""
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params["model"]


def train():
    """Train logistic regression model with MLflow tracking."""
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("Loading parameters...")
    params = load_params()
    C = params.get("C", 1.0)
    penalty = params.get("penalty", "l2")

    print(f"Training with C={C}, penalty={penalty}")

    # Set MLflow experiment
    mlflow.set_experiment("customer-churn")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("C", C)
        mlflow.log_param("penalty", penalty)

        # Train model
        model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver="lbfgs" if penalty == "l2" else "saga",
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Save model locally
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = f"{MODEL_DIR}/model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Log run ID for reference
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")

    print("Training complete!")


if __name__ == "__main__":
    train()
