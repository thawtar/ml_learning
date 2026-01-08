# Customer Churn Prediction MLOps

A minimalist MLOps application for predicting customer churn using machine learning.

## Features

- **DVC** — Data versioning and pipeline management
- **MLflow** — Experiment tracking and model registry
- **FastAPI** — REST API for model serving
- **Streamlit** — Interactive web UI
- **Docker** — Containerized deployment

## Project Structure

```
ai-build/
├── data/
│   └── customer_churn_dataset-training-master.csv
├── data_processed/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── encoders.pkl
│   └── scaler.pkl
├── models/
│   └── model.pkl
├── src/
│   ├── preprocess.py      # Data preprocessing pipeline
│   ├── train.py           # Model training with MLflow
│   ├── serve.py           # FastAPI server
│   ├── ui.py              # Streamlit UI
│   └── PIPELINE.md        # Pipeline documentation
├── dvc.yaml               # DVC pipeline definition
├── params.yaml            # Hyperparameters
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── DVC_GUIDE.md           # DVC tutorial
└── README.md              # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Pipeline with DVC

```bash
# Track data
dvc add data/customer_churn_dataset-training-master.csv

# Run preprocessing and training
dvc repro
```

### 3. Start API Server

```bash
cd src
python serve.py
```

API available at http://localhost:8000

### 4. Start UI (Optional)

```bash
streamlit run src/ui.py
```

UI available at http://localhost:8501

## Usage

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict churn |
| GET | `/docs` | Swagger UI |

### Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CustomerID": 123,
    "Age": 30,
    "Gender": "Female",
    "Tenure": 39,
    "Usage_Frequency": 14,
    "Support_Calls": 5,
    "Payment_Delay": 18,
    "Subscription_Type": "Standard",
    "Contract_Length": "Annual",
    "Total_Spend": 932.0,
    "Last_Interaction": 17
  }'
```

### Response

```json
{
  "churn": false,
  "churn_probability": 0.23
}
```

## MLflow

View experiment tracking:

```bash
mlflow ui
```

Open http://localhost:5000

### Tracked Metrics
- Accuracy
- Precision
- Recall
- F1 Score

### Tracked Parameters
- `C` — Regularization strength
- `penalty` — Regularization type (l1/l2)

## DVC Pipeline

### Stages

```
┌─────────────────┐     ┌─────────────────┐
│   preprocess    │ ──→ │     train       │
└─────────────────┘     └─────────────────┘
```

### Commands

```bash
# Run full pipeline
dvc repro

# Check status
dvc status

# View DAG
dvc dag
```

### Modify Hyperparameters

Edit `params.yaml`:
```yaml
model:
  C: 0.5        # Change this
  penalty: l2
```

Then run:
```bash
dvc repro   # Only re-runs train stage
```

## Docker

### Build

```bash
docker build -t churn-api .
```

### Run

```bash
docker run -p 8000:8000 churn-api
```

API available at http://localhost:8000

## Input Features

| Feature | Type | Description |
|---------|------|-------------|
| CustomerID | int | Customer identifier |
| Age | int | Customer age |
| Gender | string | Male / Female |
| Tenure | int | Months with company |
| Usage_Frequency | int | Usage frequency score |
| Support_Calls | int | Number of support calls |
| Payment_Delay | int | Payment delay in days |
| Subscription_Type | string | Basic / Standard / Premium |
| Contract_Length | string | Monthly / Quarterly / Annual |
| Total_Spend | float | Total amount spent |
| Last_Interaction | int | Days since last interaction |

## Model

- **Algorithm**: Logistic Regression / SVC
- **Accuracy**: ~97% on test set
- **Framework**: scikit-learn

## Documentation

- [Pipeline Details](src/PIPELINE.md)
- [DVC Guide](DVC_GUIDE.md)
