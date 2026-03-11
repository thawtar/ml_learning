# Python for ML Engineers — Practice Course

A hands-on course covering the core Python ML stack: **NumPy**, **Pandas**, **Matplotlib**, **Scikit-learn**, and **MLflow**.

Each module is a runnable `.py` file with explanations, examples, and TODO exercises.

---

## Course Structure

### Module 1 — NumPy: The Foundation
| File | Topics |
|------|--------|
| `01_numpy/01_array_creation_indexing.py` | ndarray, dtypes, reshaping, fancy indexing, boolean masks |
| `01_numpy/02_vectorized_ops_broadcasting.py` | Ufuncs, broadcasting rules, avoiding loops |
| `01_numpy/03_linear_algebra.py` | dot, matmul, inv, eig, SVD — the math behind ML |
| `01_numpy/04_random_and_statistics.py` | RNG, distributions, statistical aggregations |
| `01_numpy/05_performance_tips.py` | Memory layout, views vs copies, vectorization benchmarks |

### Module 2 — Pandas: Data Wrangling
| File | Topics |
|------|--------|
| `02_pandas/01_series_dataframe_basics.py` | Creation, indexing (loc/iloc), dtypes, info/describe |
| `02_pandas/02_data_cleaning.py` | Missing values, duplicates, type conversion, string ops |
| `02_pandas/03_groupby_aggregation.py` | split-apply-combine, agg, transform, pivot tables |
| `02_pandas/04_merge_join_concat.py` | merge, join, concat — combining datasets |
| `02_pandas/05_feature_engineering.py` | Window functions, binning, encoding categoricals, datetime |

### Module 3 — Matplotlib & Seaborn: Visualization
| File | Topics |
|------|--------|
| `03_visualization/01_matplotlib_fundamentals.py` | Figure/Axes API, line/bar/scatter, subplots, styling |
| `03_visualization/02_statistical_plots.py` | Histograms, boxplots, heatmaps, pair plots (seaborn) |
| `03_visualization/03_ml_specific_plots.py` | Confusion matrix, ROC/AUC, learning curves, feature importance |

### Module 4 — Scikit-learn: Classical ML
| File | Topics |
|------|--------|
| `04_sklearn/01_preprocessing_pipelines.py` | StandardScaler, OneHotEncoder, Pipeline, ColumnTransformer |
| `04_sklearn/02_classification.py` | LogisticRegression, RandomForest, SVM, evaluation metrics |
| `04_sklearn/03_regression.py` | LinearRegression, Ridge, Lasso, GradientBoosting, metrics |
| `04_sklearn/04_model_selection.py` | cross_val_score, GridSearchCV, RandomizedSearchCV, stratification |
| `04_sklearn/05_unsupervised.py` | KMeans, DBSCAN, PCA, t-SNE, silhouette score |

### Module 5 — MLflow: Experiment Tracking & Deployment
| File | Topics |
|------|--------|
| `05_mlflow/01_tracking_basics.py` | Runs, params, metrics, artifacts, autolog |
| `05_mlflow/02_model_registry.py` | Logging models, versioning, stage transitions |
| `05_mlflow/03_full_experiment.py` | End-to-end: load data → train → log → compare → register |

### Capstone
| File | Topics |
|------|--------|
| `06_capstone/end_to_end_ml_project.py` | Full pipeline: EDA → cleaning → feature eng → train → evaluate → track with MLflow |

---

## Prerequisites
- Python 3.10+
- `pip install numpy pandas matplotlib seaborn scikit-learn mlflow`

## How to Use
1. Read each file top-to-bottom — explanations are inline
2. Complete the `# TODO` exercises
3. Run the file: `python <filename>.py`
4. Check your work against the expected output in comments
