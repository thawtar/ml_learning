"""
Logistic Regression Baseline for Forex Prediction - Three-Class Version
========================================================================
Minimalist training script using scikit-learn for three-class classification.
Serves as a benchmark for comparing with neural network models.

Label Format: Three-class labels (DOWN=0, SIDEWAYS=1, UP=2)
"""

import numpy as np
import argparse
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
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
    print(f"  Label shape: {y_train.shape} (three-class labels)")
    
    # Verify label range
    unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    print(f"  Unique labels: {unique_labels} (0=DOWN, 1=SIDEWAYS, 2=UP)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten 3D sequences to 2D for logistic regression."""
    # X: (samples, seq_len, features) -> (samples, seq_len * features)
    return X.reshape(X.shape[0], -1)


def train_model(X_train, y_train, penalty: str = 'l2', C: float = 1.0):
    """
    Train logistic regression model for multi-class classification.
    
    Args:
        X_train: Training features (already flattened and scaled)
        y_train: Training labels (0=DOWN, 1=SIDEWAYS, 2=UP)
        penalty: Regularization type ('l1' or 'l2')
        C: Inverse regularization strength
        
    Returns:
        Trained model
    """
    print(f"\nTraining Logistic Regression (Multi-class)...")
    print(f"  Regularization: {penalty.upper()}")
    print(f"  C (inverse regularization): {C}")
    print(f"  Multi-class strategy: One-vs-Rest (OvR)")
    
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver='lbfgs' if penalty == 'l2' else 'saga',
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
        multi_class='ovr'  # One-vs-Rest for three classes
    )
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, scaler, X, y, split_name: str = "Test"):
    """
    Comprehensive model evaluation for three-class classification.
    
    Args:
        model: Trained logistic regression model
        scaler: Fitted StandardScaler
        X: Features (flattened)
        y: Labels (0=DOWN, 1=SIDEWAYS, 2=UP)
        split_name: Name of the split being evaluated
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)  # Shape: (n_samples, n_classes)
    
    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    
    # Precision, recall, F1 for each class
    prec, rec, f1, support = precision_recall_fscore_support(
        y, y_pred, average=None, zero_division=0, labels=[0, 1, 2]
    )
    
    # Macro-averaged metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y, y_pred, average='macro', zero_division=0
    )
    
    # Weighted-averaged metrics
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y, y_pred, average='weighted', zero_division=0
    )
    
    # ROC-AUC score (One-vs-Rest)
    try:
        # Binarize labels for multi-class ROC-AUC
        y_bin = label_binarize(y, classes=[0, 1, 2])
        auc_ovr = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
    except:
        auc_ovr = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
    
    results = {
        'accuracy': acc,
        'precision_per_class': prec,
        'recall_per_class': rec,
        'f1_per_class': f1,
        'support_per_class': support,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'precision_weighted': prec_weighted,
        'recall_weighted': rec_weighted,
        'f1_weighted': f1_weighted,
        'auc_ovr': auc_ovr,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y
    }
    
    # Print results
    print(f"\n{split_name} Set Evaluation:")
    print("-" * 60)
    print(f"  Overall Accuracy: {acc:.4f}")
    print(f"  Macro-averaged:")
    print(f"    Precision: {prec_macro:.4f}")
    print(f"    Recall:    {rec_macro:.4f}")
    print(f"    F1-Score:  {f1_macro:.4f}")
    print(f"  Weighted-averaged:")
    print(f"    Precision: {prec_weighted:.4f}")
    print(f"    Recall:    {rec_weighted:.4f}")
    print(f"    F1-Score:  {f1_weighted:.4f}")
    print(f"  ROC-AUC (OvR): {auc_ovr:.4f}")
    
    print(f"\n  Per-Class Metrics:")
    class_names = ['DOWN', 'SIDEWAYS', 'UP']
    for i, name in enumerate(class_names):
        print(f"    {name:8s}: Precision={prec[i]:.4f}, Recall={rec[i]:.4f}, F1={f1[i]:.4f}, Support={int(support[i])}")
    
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 DOWN  SIDEW  UP")
    print(f"  Actual DOWN    {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
    print(f"         SIDEWAYS {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
    print(f"         UP       {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    # Class distribution
    down_count = (y == 0).sum()
    sideways_count = (y == 1).sum()
    up_count = (y == 2).sum()
    total = len(y)
    
    print(f"\n  Class Distribution:")
    print(f"    DOWN (0):     {int(down_count)} ({down_count/total*100:.1f}%)")
    print(f"    SIDEWAYS (1): {int(sideways_count)} ({sideways_count/total*100:.1f}%)")
    print(f"    UP (2):       {int(up_count)} ({up_count/total*100:.1f}%)")
    
    return results


def plot_results(train_results, val_results, test_results, output_dir):
    """Generate visualization plots for three-class classification."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    class_names = ['DOWN', 'SIDEWAYS', 'UP']
    
    # 1. Overall Metrics Comparison
    plt.subplot(3, 3, 1)
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_ovr']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-OvR']
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
    plt.title('Model Performance Across Splits (Macro-averaged)')
    plt.xticks(x, metric_labels, rotation=45)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix - Test Set
    plt.subplot(3, 3, 2)
    cm = test_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # 3. Per-Class F1 Scores
    plt.subplot(3, 3, 3)
    f1_scores = test_results['f1_per_class']
    bars = plt.bar(class_names, f1_scores, color=['#ef4444', '#f59e0b', '#10b981'], alpha=0.7)
    plt.title('Per-Class F1-Scores (Test Set)')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. ROC Curves (One-vs-Rest)
    plt.subplot(3, 3, 4)
    y_test = test_results['true_labels']
    y_prob = test_results['probabilities']
    
    # Binarize labels for ROC curve
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    colors = ['#ef4444', '#f59e0b', '#10b981']
    for i, (color, name) in enumerate(zip(colors, class_names)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{name}')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves (OvR) - AUC={test_results["auc_ovr"]:.3f}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # 5. Prediction Probability Distribution
    plt.subplot(3, 3, 5)
    y_test = test_results['true_labels']
    y_prob = test_results['probabilities']
    
    for i, (color, name) in enumerate(zip(colors, class_names)):
        # Get max probability for samples where true label is this class
        mask = (y_test == i)
        if mask.sum() > 0:
            probs = y_prob[mask][:, i]  # Probability of correct class
            plt.hist(probs, bins=20, alpha=0.6, label=f'Actual {name}', color=color)
    
    plt.xlabel('Predicted Probability (Correct Class)')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Per-Class Precision and Recall
    plt.subplot(3, 3, 6)
    precision = test_results['precision_per_class']
    recall = test_results['recall_per_class']
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, precision, width, label='Precision', alpha=0.8)
    plt.bar(x + width/2, recall, width, label='Recall', alpha=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision and Recall per Class (Test)')
    plt.xticks(x, class_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 7. Class Distribution Comparison
    plt.subplot(3, 3, 7)
    
    # True distribution
    y_true = test_results['true_labels']
    true_dist = [(y_true == i).sum() for i in range(3)]
    
    # Predicted distribution
    y_pred = test_results['predictions']
    pred_dist = [(y_pred == i).sum() for i in range(3)]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, true_dist, width, label='True', alpha=0.8)
    plt.bar(x + width/2, pred_dist, width, label='Predicted', alpha=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution: True vs Predicted')
    plt.xticks(x, class_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # 8. Normalized Confusion Matrix
    plt.subplot(3, 3, 8)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=class_names,
                yticklabels=class_names,
                vmin=0, vmax=1,
                cbar_kws={'label': 'Proportion'})
    plt.title('Normalized Confusion Matrix (Test)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # 9. Classification Report Text
    plt.subplot(3, 3, 9)
    plt.axis('off')
    report = classification_report(test_results['true_labels'], 
                                   test_results['predictions'],
                                   target_names=class_names,
                                   digits=3)
    plt.text(0.1, 0.5, report, fontsize=10, family='monospace',
             verticalalignment='center')
    plt.title('Classification Report (Test Set)', fontsize=12, pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'logistic_regression_three_class_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Logistic Regression for Three-Class Forex Prediction'
    )
    parser.add_argument('--data_dir', type=str, default='data_splits_three_class', 
                       help='Data directory')
    parser.add_argument('--penalty', type=str, default='l2', 
                       choices=['l1', 'l2'], help='Regularization type')
    parser.add_argument('--C', type=float, default=1, 
                       help='Inverse regularization strength')
    parser.add_argument('--output_dir', type=str, default='models_lr_three_class', 
                       help='Model output directory')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate visualization plots')
    args = parser.parse_args()
    
    print("="*60)
    print("LOGISTIC REGRESSION - THREE-CLASS FOREX PREDICTION")
    print("="*60)
    print("Task: Predict DOWN (0), SIDEWAYS (1), or UP (2) movement")
    print("="*60)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_dir)
    
    # Check label distribution
    print(f"\nLabel Distribution:")
    for name, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        down_pct = (labels == 0).sum() / len(labels) * 100
        sideways_pct = (labels == 1).sum() / len(labels) * 100
        up_pct = (labels == 2).sum() / len(labels) * 100
        print(f"  {name:5s}: DOWN {down_pct:.1f}% | SIDEWAYS {sideways_pct:.1f}% | UP {up_pct:.1f}%")
    
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
    val_results = evaluate_model(model, scaler, X_val_flat, y_val, "Validation")
    test_results = evaluate_model(model, scaler, X_test_flat, y_test, "Test")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'logistic_regression_three_class_model.joblib')
    joblib.dump({'model': model, 'scaler': scaler}, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(args.output_dir, f'results_three_class_{timestamp}.txt')
    
    with open(results_file, 'w') as f:
        f.write("LOGISTIC REGRESSION - THREE-CLASS FOREX PREDICTION\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Regularization: {args.penalty.upper()}, C={args.C}\n")
        f.write(f"Features: {X_train_flat.shape[1]}\n")
        f.write(f"Classes: 3 (DOWN=0, SIDEWAYS=1, UP=2)\n\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-"*60 + "\n")
        for split_name, results in [('Train', train_results), 
                                     ('Val', val_results), 
                                     ('Test', test_results)]:
            f.write(f"\n{split_name} Set:\n")
            f.write(f"  Accuracy:          {results['accuracy']:.4f}\n")
            f.write(f"  Precision (macro): {results['precision_macro']:.4f}\n")
            f.write(f"  Recall (macro):    {results['recall_macro']:.4f}\n")
            f.write(f"  F1-Score (macro):  {results['f1_macro']:.4f}\n")
            f.write(f"  ROC-AUC (OvR):     {results['auc_ovr']:.4f}\n")
            
            f.write(f"\n  Per-Class Metrics:\n")
            class_names = ['DOWN', 'SIDEWAYS', 'UP']
            for i, name in enumerate(class_names):
                f.write(f"    {name:8s}: P={results['precision_per_class'][i]:.4f}, "
                       f"R={results['recall_per_class'][i]:.4f}, "
                       f"F1={results['f1_per_class'][i]:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("FEATURE IMPORTANCE (Top 20 per class)\n")
        f.write("-"*60 + "\n")
        
        # Get feature importance (coefficients) for each class
        coef = model.coef_  # Shape: (n_classes, n_features)
        
        for class_idx, class_name in enumerate(['DOWN', 'SIDEWAYS', 'UP']):
            f.write(f"\n{class_name} (Class {class_idx}):\n")
            feature_importance = np.abs(coef[class_idx])
            top_indices = np.argsort(feature_importance)[-20:][::-1]
            
            for idx in top_indices:
                f.write(f"  Feature {idx:4d}: {coef[class_idx][idx]:+.6f}\n")
    
    print(f"Results saved to: {results_file}")
    
    # Generate plots
    if args.plot:
        print(f"\nGenerating visualization plots...")
        plot_results(train_results, val_results, test_results, args.output_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Test Accuracy:         {test_results['accuracy']:.4f}")
    print(f"Test F1-Score (macro): {test_results['f1_macro']:.4f}")
    print(f"Test ROC-AUC (OvR):    {test_results['auc_ovr']:.4f}")
    print("\nPer-Class F1-Scores (Test):")
    class_names = ['DOWN', 'SIDEWAYS', 'UP']
    for i, name in enumerate(class_names):
        f1 = test_results['f1_per_class'][i]
        print(f"  {name:8s}: {f1:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()