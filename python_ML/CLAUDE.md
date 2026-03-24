# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A hands-on Python ML training course with 6 progressive modules: NumPy, Pandas, Matplotlib/Seaborn, Scikit-learn, MLflow, and a Capstone project. Each module consists of runnable `.py` files with inline explanations and TODO exercises.

## Running Code

```bash
# Run any lesson file directly
python 01_numpy/01_array_creation_indexing.py

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn mlflow
```

There are no tests, linting, or build systems configured. Files are standalone scripts, not importable modules.

## Architecture

- **Module progression is linear:** NumPy → Pandas → Matplotlib → Scikit-learn → MLflow → Capstone
- **File naming convention:** `NN_module/NN_topic_name.py` (e.g., `04_sklearn/02_classification.py`)
- **Data sources:** All examples use `sklearn.datasets` built-in datasets (Iris, breast_cancer, California Housing, synthetic) — no external data files
- **Capstone** (`06_capstone/end_to_end_ml_project.py`) ties all modules together with TODO exercises referencing prior lessons
- **Jupyter notebooks** at root (`python_ML_1_*.ipynb`) cover early NumPy topics

## Code Style

- Each file is a self-contained lesson with section headers using `═` visual separators
- Heavy inline comments explaining ML concepts — preserve this pedagogical style when editing
- Python 3.10+ required
