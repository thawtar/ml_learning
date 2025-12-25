"""
XGBoost Baseline for Forex Prediction - Single Label Version
==========================================================================
Minimalist training script using scikit-learn for binary classification.
Serves as a benchmark for comparing with neural network models.

Label Format: Single binary label (UP=1, DOWN=0)
"""

import numpy as np
import argparse
import os
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_dir: str = 'data_splits'):
    """Load training data from npz files."""
    print(f"Loading data from {data_dir}...")
    
    train = np.load(os.path.join(data_dir, 'train_data.npz'))
    val = np.load(os.path.join(data_dir, 'val_data.npz'))
    test = np.load(os.path.join(data_dir, 'test_data.npz'))
    
    X_train, y_train = train['X'], train['y']
    X_val, y_val = val['X'], val['y']
    X_test, y_test = test['X'], test['y']
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Sequence shape: {X_train.shape[1:]} (seq_len, features)")
    print(f"  Label shape: {y_train.shape} (single binary label)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten 3D sequences to 2D for XGBoost."""
    # X: (samples, seq_len, features) -> (samples, seq_len * features)
    return X.reshape(X.shape[0], -1)


def train_model(X_train, y_train, penalty: str = 'l2', C: float = 1.0):
    """
    Train XGBoost model.
    
    Args:
        X_train: Training features (already flattened and scaled)
        y_train: Training labels (binary)
        penalty: Regularization type ('l1' or 'l2')
        C: Inverse regularization strength
        
    Returns:
        Trained model
    """
    print(f"\nTraining XGBoost...")
    print(f"  Regularization: {penalty.upper()}")
    print(f"  C (inverse regularization): {C}")
    
    
    # model = XGBClassifier(
    #     objective='binary:logistic',
    #     learning_rate=0.01,
    #     eval_metric='logloss',
    #     random_state=42
    # )
    model = XGBClassifier(
        max_depth=2,
        n_estimators=50,
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=5.0
)
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, scaler, X, y, split_name: str = "Test"):
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        X: Features (flattened)
        y: Labels
        split_name: Name of the split being evaluated
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]  # Probability of UP class
    
    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='binary', zero_division=0
    )
    
    # ROC-AUC score
    try:
        auc = roc_auc_score(y, y_prob)
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_prob
    }
    
    # Print results
    print(f"\n{split_name} Set Evaluation:")
    print("-" * 60)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              DOWN  UP")
    print(f"  Actual DOWN  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"         UP    {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Class distribution
    up_count = y.sum()
    down_count = len(y) - up_count
    print(f"\n  Class Distribution:")
    print(f"    DOWN (0): {down_count} ({down_count/len(y)*100:.1f}%)")
    print(f"    UP (1):   {int(up_count)} ({up_count/len(y)*100:.1f}%)")
    
    return results


def plot_results(train_results, val_results, test_results, output_dir):
    """Generate visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Metrics Comparison
    plt.subplot(2, 3, 1)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    train_vals = [train_results[m] for m in metrics]
    val_vals = [val_results[m] for m in metrics]
    test_vals = [test_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.bar(x - width, train_vals, width, label='Train', alpha=0.8)
    plt.bar(x, val_vals, width, label='Val', alpha=0.8)
    plt.bar(x + width, test_vals, width, label='Test', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Across Splits')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix - Test Set
    plt.subplot(2, 3, 2)
    cm = test_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['DOWN', 'UP'],
                yticklabels=['DOWN', 'UP'])
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # 3. ROC Curve
    plt.subplot(2, 3, 3)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(test_results['true_labels'], 
                            test_results['probabilities'])
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={test_results["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Prediction Distribution
    plt.subplot(2, 3, 4)
    test_probs = test_results['probabilities']
    test_labels = test_results['true_labels']
    
    plt.hist(test_probs[test_labels == 0], bins=30, alpha=0.6, 
             label='Actual DOWN', color='red')
    plt.hist(test_probs[test_labels == 1], bins=30, alpha=0.6, 
             label='Actual UP', color='green')
    plt.xlabel('Predicted Probability (UP)')
    plt.ylabel('Frequency')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Metrics by Threshold
    plt.subplot(2, 3, 5)
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        pred = (test_probs >= thresh).astype(int)
        acc = accuracy_score(test_labels, pred)
        prec = precision_recall_fscore_support(test_labels, pred, 
                                               average='binary', 
                                               zero_division=0)[0]
        rec = precision_recall_fscore_support(test_labels, pred, 
                                              average='binary', 
                                              zero_division=0)[1]
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
    
    plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.axvline(0.5, color='k', linestyle='--', alpha=0.3, label='Default (0.5)')
    plt.xlabel('Prediction Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Decision Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Classification Report Text
    plt.subplot(2, 3, 6)
    plt.axis('off')
    report = classification_report(test_results['true_labels'], 
                                   test_results['predictions'],
                                   target_names=['DOWN', 'UP'])
    plt.text(0.1, 0.5, report, fontsize=10, family='monospace',
             verticalalignment='center')
    plt.title('Classification Report (Test Set)', fontsize=12, pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'xgboost_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train XGBoost for Binary Forex Prediction'
    )
    parser.add_argument('--data_dir', type=str, default='data_splits', 
                       help='Data directory')
    parser.add_argument('--penalty', type=str, default='l2', 
                       choices=['l1', 'l2'], help='Regularization type')
    parser.add_argument('--C', type=float, default=1.0, 
                       help='Inverse regularization strength')
    parser.add_argument('--output_dir', type=str, default='models_lr', 
                       help='Model output directory')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate visualization plots')
    args = parser.parse_args()
    
    print("="*60)
    print("XGBOOST - FOREX BINARY PREDICTION")
    print("="*60)
    print("Task: Predict UP (1) or DOWN (0) movement")
    print("="*60)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_dir)
    
    # Check label distribution
    print(f"\nLabel Distribution:")
    for name, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        up_pct = (labels.sum() / len(labels)) * 100
        print(f"  {name:5s}: UP {up_pct:.1f}% | DOWN {100-up_pct:.1f}%")
    
    # Flatten sequences
    print(f"\nFlattening sequences...")
    X_train_flat = flatten_sequences(X_train)
    X_val_flat = flatten_sequences(X_val)
    X_test_flat = flatten_sequences(X_test)
    print(f"  Flattened feature dimension: {X_train_flat.shape[1]}")
    
    # Scale features
    print(f"\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Train model
    model = train_model(X_train_scaled, y_train, 
                       penalty=args.penalty, C=args.C)
    
    # Evaluate on all splits
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    train_results = evaluate_model(model, scaler, X_train_flat, y_train, "Train")
    train_results['true_labels'] = y_train
    
    val_results = evaluate_model(model, scaler, X_val_flat, y_val, "Validation")
    val_results['true_labels'] = y_val
    
    test_results = evaluate_model(model, scaler, X_test_flat, y_test, "Test")
    test_results['true_labels'] = y_test
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'xgboost_model.joblib')
    joblib.dump({'model': model, 'scaler': scaler}, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(args.output_dir, f'results_{timestamp}.txt')
    
    with open(results_file, 'w') as f:
        f.write("XGBOOST - BINARY FOREX PREDICTION\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Regularization: {args.penalty.upper()}, C={args.C}\n")
        f.write(f"Features: {X_train_flat.shape[1]}\n\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-"*60 + "\n")
        for split_name, results in [('Train', train_results), 
                                     ('Val', val_results), 
                                     ('Test', test_results)]:
            f.write(f"\n{split_name} Set:\n")
            f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall:    {results['recall']:.4f}\n")
            f.write(f"  F1-Score:  {results['f1']:.4f}\n")
            f.write(f"  ROC-AUC:   {results['auc']:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("FEATURE IMPORTANCE (Top 20)\n")
        f.write("-"*60 + "\n")
        
        # # Get feature importance (coefficients)
        # coef = model.coef_[0]
        # feature_importance = np.abs(coef)
        # top_indices = np.argsort(feature_importance)[-20:][::-1]
        
        # for idx in top_indices:
        #     f.write(f"  Feature {idx:4d}: {coef[idx]:+.6f}\n")
    
    print(f"Results saved to: {results_file}")
    
    # Generate plots
    if args.plot:
        print(f"\nGenerating visualization plots...")
        plot_results(train_results, val_results, test_results, args.output_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1-Score: {test_results['f1']:.4f}")
    print(f"Test ROC-AUC:  {test_results['auc']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()