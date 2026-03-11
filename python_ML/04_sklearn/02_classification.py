"""
Module 4 — Lesson 2: Classification
=====================================
The classification workflow: train → predict → evaluate.
Covers the most common classifiers, metrics, and evaluation patterns.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ── Classifiers ──
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ── Metrics ──
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, log_loss,
)


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=12,
    n_redundant=4, n_classes=2, random_state=42, flip_y=0.05,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Class distribution: {np.bincount(y_train)}")


# ══════════════════════════════════════════════════════════════════════
# 1. MODEL ZOO — Train & Compare
# ══════════════════════════════════════════════════════════════════════
print("\n── Model comparison ──")

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN (k=5)":           make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)":           make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Log Loss": log_loss(y_test, y_prob),
    })

results_df = pd.DataFrame(results).round(4)
print(results_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
# 2. DETAILED EVALUATION
# ══════════════════════════════════════════════════════════════════════
print("\n── Detailed evaluation (Random Forest) ──")

rf = models["Random Forest"]
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# Classification report (precision, recall, f1 per class)
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


# ══════════════════════════════════════════════════════════════════════
# 3. METRICS EXPLAINED
# ══════════════════════════════════════════════════════════════════════
"""
WHEN TO USE WHAT:

               │ Balanced data  │ Imbalanced data
───────────────┼────────────────┼─────────────────
Default metric │ Accuracy       │ ⚠ Misleading!
               │                │
Class-wise     │ Precision      │ Precision (when FP is costly)
               │ Recall         │ Recall (when FN is costly)
               │ F1 (harmonic)  │ F1 (balanced tradeoff)
               │                │
Ranking        │ AUC-ROC        │ AUC-PR (precision-recall)
               │                │
Probabilistic  │ Log Loss       │ Brier Score

Examples:
  - Spam filter: high Precision (don't trash real email)
  - Cancer screening: high Recall (don't miss cancer)
  - Search ranking: AUC-ROC (rank relevant docs higher)
  - Imbalanced fraud: F1 or AUC-PR
"""


# ══════════════════════════════════════════════════════════════════════
# 4. HANDLING IMBALANCED CLASSES
# ══════════════════════════════════════════════════════════════════════
print("\n── Imbalanced classes ──")

# Create imbalanced dataset (95% negative, 5% positive)
X_imb, y_imb = make_classification(
    n_samples=2000, n_features=20, n_informative=10,
    weights=[0.95, 0.05], random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X_imb, y_imb, test_size=0.3,
                                            random_state=42, stratify=y_imb)
print(f"Class distribution: {np.bincount(y_tr)}")

# Naive model → high accuracy but terrible at finding positives
naive = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(f"\nNaive - Accuracy: {naive.score(X_te, y_te):.3f}")
print(f"Naive - Recall: {recall_score(y_te, naive.predict(X_te)):.3f}")

# Solution 1: class_weight="balanced"
balanced = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr, y_tr)
print(f"\nBalanced - Accuracy: {balanced.score(X_te, y_te):.3f}")
print(f"Balanced - Recall: {recall_score(y_te, balanced.predict(X_te)):.3f}")
print(f"Balanced - F1: {f1_score(y_te, balanced.predict(X_te)):.3f}")

# Solution 2: Use F1/AUC as the optimization metric instead of accuracy
cv_f1 = cross_val_score(balanced, X_imb, y_imb, cv=5, scoring="f1")
print(f"\nCV F1 (balanced): {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")


# ══════════════════════════════════════════════════════════════════════
# 5. PROBABILITY CALIBRATION & THRESHOLD TUNING
# ══════════════════════════════════════════════════════════════════════
print("\n── Threshold tuning ──")

# Default threshold is 0.5. Often suboptimal for imbalanced problems.
y_prob = balanced.predict_proba(X_te)[:, 1]

# Find best threshold by F1
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_te, y_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.3f}")
print(f"Best F1 at this threshold: {f1_scores[best_idx]:.3f}")

# Apply custom threshold
y_custom = (y_prob >= best_threshold).astype(int)
print(f"Default threshold F1:  {f1_score(y_te, balanced.predict(X_te)):.3f}")
print(f"Optimal threshold F1:  {f1_score(y_te, y_custom):.3f}")


# ══════════════════════════════════════════════════════════════════════
# 6. ENSEMBLE: VOTING CLASSIFIER
# ══════════════════════════════════════════════════════════════════════
print("\n── Voting ensemble ──")

ensemble = VotingClassifier(
    estimators=[
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ],
    voting="soft",   # use probabilities (better than "hard" majority vote)
)

cv_ensemble = cross_val_score(ensemble, X, y, cv=5, scoring="accuracy")
print(f"Ensemble CV: {cv_ensemble.mean():.3f} ± {cv_ensemble.std():.3f}")


# ══════════════════════════════════════════════════════════════════════
# EXERCISES
# ══════════════════════════════════════════════════════════════════════
"""
Exercise 2.1: Load the Breast Cancer dataset. Train and compare at least
    4 classifiers. Report accuracy, F1, and AUC for each. Which is best?

Exercise 2.2: Create an imbalanced dataset (90/10 split). Compare:
    - LogisticRegression (default)
    - LogisticRegression (class_weight="balanced")
    - RandomForest (class_weight="balanced")
    Report F1 and recall for the minority class.

Exercise 2.3: Implement a function that finds the optimal classification
    threshold by maximizing F1 score. Plot F1 vs threshold.

Exercise 2.4: Build a VotingClassifier with 3+ models. Compare its
    performance to each individual model. Does the ensemble always win?

Exercise 2.5: Research: What is the difference between "soft" and "hard"
    voting? When would you prefer one over the other?
"""
