# Thaw's MLOps Self Study Repository

This is Thaw's MLOps self study repository. Some of the python codes and scripts (including the following README.md) are created using AI but most of the code are manually written, and checked and edited by human.

## Contents

1. [ML Learning Notebooks](#1-ml-learning-notebooks)
2. [Customer Churning Project](#2-customer-churning-project)
3. [Project Template](#3-project-template)

---

## 1. ML Learning Notebooks

The `ml_learning_notebooks` folder includes Jupyter notebooks on various traditional ML algorithms. Some algorithms like logistic regression and perceptron were created from scratch while Scikit-Learn was used for most other cases. Some notebooks contain examples from the book "Machine Learning with Pytorch and Scikit-Learn".

### Notebooks Overview

| Notebook | Topic |
|----------|-------|
| 00_1_technical_indicators | Technical indicators for financial data analysis |
| 01_logisticRegression | Logistic regression fundamentals |
| 02_01_perceptron | Perceptron algorithm implementation from scratch |
| 03_01_logisticRegressionGD | Logistic regression with gradient descent |
| 03_02_SVM | Support Vector Machines |
| 03_03_decisionTree_kNN | Decision Trees and k-Nearest Neighbors |
| 03_kaggle_imdb | IMDB dataset analysis (Kaggle example) |
| 04_01_missing_data | Handling missing data in datasets |
| 04_02_random_forest | Random Forest classifier |
| 04_02_sequential_backward_selection | Feature selection techniques |
| 06_01_pipeline | Building ML pipelines with Scikit-Learn |
| 06_02_bias_variance_curve | Understanding bias-variance tradeoff |
| 06_03_hyperparameter_search | Grid search and hyperparameter tuning |

---

## 2. Customer Churning Project

The `customer_churning` folder is a simple full stack MLOps project Thaw created to learn the MLOps workflow. This project demonstrates a complete machine learning pipeline from data preprocessing to model deployment.

### Project Structure

```
customer_churning/
├── data/              # Raw dataset
├── data_processed/    # Processed/transformed data
├── models/            # Saved trained models
├── notebooks/         # Exploration notebooks
├── src/
│   ├── preprocess.py  # Data preprocessing pipeline
│   ├── train.py       # Model training script
│   ├── serve.py       # FastAPI model serving endpoint
│   └── ui.py          # Streamlit UI for predictions
├── Dockerfile         # Container for ML pipeline
├── Dockerfile.ui      # Container for Streamlit UI
├── docker-compose.yml # Multi-container deployment
├── dvc.yaml           # DVC pipeline configuration
└── requirements.txt   # Python dependencies
```

### Tech Stack

- **Data Processing**: Pandas, Scikit-Learn
- **Pipeline Management**: DVC (Data Version Control)
- **Experiment Tracking**: MLflow
- **Model Serving**: FastAPI with Uvicorn
- **User Interface**: Streamlit
- **Containerization**: Docker & Docker Compose

### How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Run DVC pipeline: `dvc repro` (preprocess → train)
3. Start the API server: `python src/serve.py`
4. Start the UI: `streamlit run src/ui.py`

Or use Docker Compose for containerized deployment:
```bash
docker-compose up --build
```

### Model Performance

For simplicity, Logistic Regression was used and a satisfactory accuracy of around 97% was attained on the test set. You can change the ML algorithm to more complex ones like SVM, XGBoost or MLPs by modifying `train.py`.

---

## 3. Project Template

The `project_template` folder is a collection of useful files that can be used in future projects. Most of the files are taken from Customer Churning project.

### Included Files

- `dvc.yaml` - DVC pipeline configuration template
- `params.yaml` - Hyperparameter configuration file
- `docker-compose.yml` - Multi-container deployment setup
- `Dockerfile` - Container build instructions
- `requirements.txt` - Common Python dependencies for ML projects
- `template.py` - Project directory structure generator
- `src/` - Source code directory structure

This template provides a quick starting point for new MLOps projects with best practices for reproducibility and deployment already configured.