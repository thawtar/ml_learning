"""FastAPI model serving application."""
from contextlib import asynccontextmanager
from typing import Literal
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "../models/model.pkl"
ENCODERS_PATH = "../data_processed/encoders.pkl"
SCALER_PATH = "../data_processed/scaler.pkl"

# Global model and preprocessors
model = None
encoders = None
scaler = None


class CustomerData(BaseModel):
    """Input schema for customer data."""
    CustomerID: int
    Age: int
    Gender: Literal["Male", "Female"]
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: Literal["Basic", "Standard", "Premium"]
    Contract_Length: Literal["Monthly", "Quarterly", "Annual"]
    Total_Spend: float
    Last_Interaction: int


class PredictionResponse(BaseModel):
    """Output schema for prediction response."""
    churn: bool
    churn_probability: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessors at startup."""
    global model, encoders, scaler
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and preprocessors loaded successfully")
    except FileNotFoundError as e:
        print(f"Warning: Could not load model or preprocessors: {e}")
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using logistic regression",
    version="1.0.0",
    lifespan=lifespan
)


def preprocess_input(data: CustomerData) -> pd.DataFrame:
    """Preprocess input data for prediction."""
    # Convert to DataFrame
    df = pd.DataFrame([data.model_dump()])

    # Rename columns: underscores to spaces (to match training data)
    df.columns = df.columns.str.replace('_', ' ')

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))

    # Reorder columns to match training data (before scaling)
    expected_cols = ["CustomerID", "Age", "Gender", "Tenure", "Usage Frequency",
                     "Support Calls", "Payment Delay", "Subscription Type",
                     "Contract Length", "Total Spend", "Last Interaction"]
    df = df[expected_cols]

    # Scale ALL columns (scaler was fit on all columns after encoding)
    df[expected_cols] = scaler.transform(df[expected_cols])

    return df


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """Predict customer churn."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run training first."
        )

    try:
        # Preprocess input
        X = preprocess_input(customer)

        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        return PredictionResponse(
            churn=bool(prediction),
            churn_probability=float(probability)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
