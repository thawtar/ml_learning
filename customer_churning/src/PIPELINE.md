# Customer Churn Prediction Pipeline

## Overview

This pipeline predicts customer churn using logistic regression. It consists of three main components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  preprocess.py  │ --> │    train.py     │ --> │    serve.py     │
│  (Data Prep)    │     │  (Model Train)  │     │  (API Serving)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Pipeline Scripts

### 1. preprocess.py

**Purpose:** Prepare raw data for model training.

**Input:**
- `data/customer_churn_data.csv` - Raw customer data

**Output:**
- `data_processed/X_train.csv` - Training features
- `data_processed/X_test.csv` - Test features
- `data_processed/y_train.csv` - Training labels
- `data_processed/y_test.csv` - Test labels
- `data_processed/encoders.pkl` - Fitted label encoders
- `data_processed/scaler.pkl` - Fitted standard scaler

**Processing Steps:**
1. Load raw CSV data
2. Drop `customerID` column
3. Convert `TotalCharges` to numeric
4. Split into train/test (80/20, stratified)
5. Impute missing values (median from training data)
6. Encode categorical features (fit on train only)
7. Scale numerical features (fit on train only)

**Run:**
```bash
python src/preprocess.py
```

---

### 2. train.py

**Purpose:** Train logistic regression model with MLflow tracking.

**Input:**
- `data_processed/X_train.csv`
- `data_processed/X_test.csv`
- `data_processed/y_train.csv`
- `data_processed/y_test.csv`
- `params.yaml` - Hyperparameters

**Output:**
- `models/model.pkl` - Trained model
- MLflow experiment logs

**Hyperparameters (from params.yaml):**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `C` | Inverse regularization strength | 1.0 |
| `penalty` | Regularization type (l1/l2) | l2 |

**Metrics Logged:**
- Accuracy
- Precision
- Recall
- F1 Score

**Run:**
```bash
python src/train.py
```

**View MLflow UI:**
```bash
mlflow ui
# Open http://localhost:5000
```

---

### 3. serve.py

**Purpose:** Serve model predictions via REST API.

**Loads at Startup:**
- `models/model.pkl`
- `data_processed/encoders.pkl`
- `data_processed/scaler.pkl`

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check, returns model load status |
| POST | `/predict` | Predict churn for a customer |

**Request Schema (POST /predict):**
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 50.0,
  "TotalCharges": 600.0
}
```

**Response Schema:**
```json
{
  "churn": false,
  "churn_probability": 0.23
}
```

**Run:**
```bash
python src/serve.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

---

## DVC Pipeline (dvc.yaml)

DVC manages the ML pipeline with two stages:

### Stage: preprocess

```yaml
preprocess:
  cmd: python src/preprocess.py
  deps:
    - data/customer_churn_data.csv
    - src/preprocess.py
  outs:
    - data_processed/X_train.csv
    - data_processed/X_test.csv
    - data_processed/y_train.csv
    - data_processed/y_test.csv
    - data_processed/encoders.pkl
    - data_processed/scaler.pkl
```

- **deps:** Files that trigger re-run when changed
- **outs:** Files tracked by DVC

### Stage: train

```yaml
train:
  cmd: python src/train.py
  deps:
    - data_processed/X_train.csv
    - data_processed/X_test.csv
    - data_processed/y_train.csv
    - data_processed/y_test.csv
    - src/train.py
  params:
    - model.C
    - model.penalty
  outs:
    - models/model.pkl
```

- **params:** Values from `params.yaml` that trigger re-run when changed

### Pipeline Commands

```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro preprocess
dvc repro train

# View pipeline DAG
dvc dag

# Check pipeline status
dvc status
```

---

## Directory Structure

```
ai-build/
├── data/
│   └── customer_churn_data.csv    # Raw data
├── data_processed/                 # DVC tracked outputs
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── encoders.pkl
│   └── scaler.pkl
├── models/
│   └── model.pkl                   # Trained model
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── serve.py
│   └── PIPELINE.md                 # This file
├── dvc.yaml                        # Pipeline definition
├── params.yaml                     # Hyperparameters
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline
dvc repro

# 3. Start API server
python src/serve.py

# 4. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,...}'
```
