"""Data preprocessing pipeline for customer churn prediction."""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

DATA_PATH = "data/customer_churn_data.csv"
OUTPUT_DIR = "data_processed"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    """Load raw data from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data: handle missing values and fix data types."""
    df = df.copy()

    # Drop customerID - not useful for prediction
    df = df.drop(columns=["customerID"])

    # TotalCharges has some blank strings - convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with median
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical features using LabelEncoder."""
    df = df.copy()
    encoders = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target from categorical columns if present
    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def scale_features(X: pd.DataFrame, scaler: StandardScaler = None) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical features using StandardScaler."""
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    if scaler is None:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    else:
        X[numerical_cols] = scaler.transform(X[numerical_cols])

    return X, scaler


def preprocess() -> None:
    """Run full preprocessing pipeline."""
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    print("Cleaning data...")
    df = clean_data(df)

    print("Encoding categorical features...")
    df, encoders = encode_features(df)

    # Encode target variable
    target_encoder = LabelEncoder()
    df["Churn"] = target_encoder.fit_transform(df["Churn"])
    encoders["Churn"] = target_encoder

    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print("Scaling numerical features...")
    X, scaler = scale_features(X)

    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save processed data
    print(f"Saving processed data to {OUTPUT_DIR}/...")
    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

    # Save encoders and scaler for serving
    joblib.dump(encoders, f"{OUTPUT_DIR}/encoders.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print("Preprocessing complete!")


if __name__ == "__main__":
    preprocess()
