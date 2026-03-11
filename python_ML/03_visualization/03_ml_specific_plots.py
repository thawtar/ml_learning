"""
Module 3 — Lesson 3: ML-Specific Plots
========================================
These are the plots you'll create in every ML project:
confusion matrix, ROC curve, learning curves, feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
    classification_report,
)

rng = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════════════
# GENERATE A CLASSIFICATION DATASET
# ══════════════════════════════════════════════════════════════════════
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_redundant=5, n_classes=2, random_state=42
)
feature_names = [f"feat_{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train models
lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)


# ══════════════════════════════════════════════════════════════════════
# 1. CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, model, name in [(axes[0], lr, "Logistic Regression"),
                          (axes[1], rf, "Random Forest")]:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"{name}\n{classification_report(y_test, y_pred, output_dict=True)['accuracy']:.3f} accuracy")

fig.suptitle("Confusion Matrices", fontsize=14)
fig.tight_layout()
fig.savefig("ml_01_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved: ml_01_confusion_matrix.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 2. ROC CURVE & AUC
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 6))

for model, name, color in [(lr, "Logistic Regression", "steelblue"),
                             (rf, "Random Forest", "coral")]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

fig.savefig("ml_02_roc_curve.png", dpi=150, bbox_inches="tight")
print("Saved: ml_02_roc_curve.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 3. PRECISION-RECALL CURVE
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 6))

for model, name, color in [(lr, "Logistic Regression", "steelblue"),
                             (rf, "Random Forest", "coral")]:
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    ax.plot(recall, precision, color=color, linewidth=2, label=name)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend()
ax.grid(True, alpha=0.3)

fig.savefig("ml_03_precision_recall.png", dpi=150, bbox_inches="tight")
print("Saved: ml_03_precision_recall.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest feature importance (Gini importance)
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)
top_k = 10

axes[0].barh(range(top_k), importances[sorted_idx[-top_k:]],
             color="steelblue", edgecolor="black", linewidth=0.5)
axes[0].set_yticks(range(top_k))
axes[0].set_yticklabels([feature_names[i] for i in sorted_idx[-top_k:]])
axes[0].set_xlabel("Gini Importance")
axes[0].set_title("RF Feature Importance (Top 10)")

# Logistic Regression coefficients
coefs = np.abs(lr.coef_[0])
sorted_idx = np.argsort(coefs)

axes[1].barh(range(top_k), coefs[sorted_idx[-top_k:]],
             color="coral", edgecolor="black", linewidth=0.5)
axes[1].set_yticks(range(top_k))
axes[1].set_yticklabels([feature_names[i] for i in sorted_idx[-top_k:]])
axes[1].set_xlabel("|Coefficient|")
axes[1].set_title("LR Feature Importance (Top 10)")

fig.tight_layout()
fig.savefig("ml_04_feature_importance.png", dpi=150, bbox_inches="tight")
print("Saved: ml_04_feature_importance.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 5. LEARNING CURVES
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 6))

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.1, color="steelblue")
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.1, color="coral")
ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Training")
ax.plot(train_sizes, val_mean, "o-", color="coral", label="Validation")

ax.set_xlabel("Training Set Size")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve — Random Forest")
ax.legend()
ax.grid(True, alpha=0.3)

fig.savefig("ml_05_learning_curve.png", dpi=150, bbox_inches="tight")
print("Saved: ml_05_learning_curve.png")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# 6. PREDICTION DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 5))

y_prob = rf.predict_proba(X_test)[:, 1]

ax.hist(y_prob[y_test == 0], bins=30, alpha=0.6, label="Negative", color="steelblue")
ax.hist(y_prob[y_test == 1], bins=30, alpha=0.6, label="Positive", color="coral")
ax.axvline(0.5, color="black", linestyle="--", label="Threshold = 0.5")

ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Count")
ax.set_title("Prediction Distribution by True Class")
ax.legend()

fig.savefig("ml_06_prediction_dist.png", dpi=150, bbox_inches="tight")
print("Saved: ml_06_prediction_dist.png")
plt.close()

print("\nAll ML plots saved.")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 3.1: Train a classifier on the Breast Cancer dataset
    (sklearn.datasets.load_breast_cancer). Create:
    - Confusion matrix
    - ROC curve
    - Feature importance (top 10)

Exercise 3.2: Create a "threshold analysis" plot:
    x-axis = classification threshold (0 to 1)
    y-axis = precision and recall (two lines)
    Mark the point where they intersect (break-even point).

Exercise 3.3: Compare 3+ models on the same ROC plot.
    Add a table below the plot showing each model's AUC, accuracy,
    and F1 score.

Exercise 3.4: Create a learning curve comparison: plot learning curves
    for Logistic Regression, Random Forest, and SVM on the same figure
    (3 subplots or overlaid).
"""
