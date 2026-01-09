"""Training script with MLflow experiment tracking."""
import os
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

DATA_DIR = "../data_processed"
MODEL_DIR = "../models"
PARAMS_FILE = "../params.yaml"
RANDOM_STATE = 42

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
    pass


if __name__ == "__main__":
    train()
