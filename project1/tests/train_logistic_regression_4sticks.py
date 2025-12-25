"""
Logistic Regression Baseline for Forex Prediction
==================================================
Minimalist training script using scikit-learn.
Serves as a benchmark for comparing with neural network models.
"""

import numpy as np
import argparse
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib


def load_data(data_dir: str = 'data_splits'):
    """Load training data from npz files."""
    train = np.load(os.path.join(data_dir, 'train_data.npz'))
    val = np.load(os.path.join(data_dir, 'val_data.npz'))
    test = np.load(os.path.join(data_dir, 'test_data.npz'))
    
    return (train['X'], train['y'], val['X'], val['y'], test['X'], test['y'])


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten 3D sequences to 2D for logistic regression."""
    # X: (samples, seq_len, features) -> (samples, seq_len * features)
    return X.reshape(X.shape[0], -1)


def train_and_evaluate(X_train, y_train, X_test, y_test, 
                       position: int, penalty: str = 'l2', C: float = 1.0):
    """Train logistic regression for a single future position."""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        solver='lbfgs' if penalty == 'l2' else 'saga',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    return {
        'model': model,
        'scaler': scaler,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'probabilities': y_prob
    }


def main():
    parser = argparse.ArgumentParser(description='Train Logistic Regression for Forex Prediction')
    parser.add_argument('--data_dir', type=str, default='data_splits', help='Data directory')
    parser.add_argument('--penalty', type=str, default='l2', choices=['l1', 'l2'], help='Regularization')
    parser.add_argument('--C', type=float, default=1.0, help='Inverse regularization strength')
    parser.add_argument('--output_dir', type=str, default='models_lr', help='Model output directory')
    args = parser.parse_args()
    
    print("="*60)
    print("LOGISTIC REGRESSION - FOREX PREDICTION BASELINE")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.data_dir)
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Sequence shape: {X_train.shape[1:]} (seq_len, features)")
    print(f"  Future window: {y_train.shape[1]} candles")
    
    # Flatten sequences
    X_train_flat = flatten_sequences(X_train)
    X_val_flat = flatten_sequences(X_val)
    X_test_flat = flatten_sequences(X_test)
    
    print(f"  Flattened features: {X_train_flat.shape[1]}")
    
    # Train model for each future position
    os.makedirs(args.output_dir, exist_ok=True)
    future_window = y_train.shape[1]
    results = []
    
    print(f"\nTraining with {args.penalty.upper()} regularization (C={args.C})...")
    print("-"*60)
    
    for pos in range(future_window):
        result = train_and_evaluate(
            X_train_flat, y_train[:, pos],
            X_test_flat, y_test[:, pos],
            position=pos,
            penalty=args.penalty,
            C=args.C
        )
        results.append(result)
        
        print(f"Candle {pos+1}: Acc={result['accuracy']:.3f}  "
              f"Prec={result['precision']:.3f}  Rec={result['recall']:.3f}  "
              f"F1={result['f1']:.3f}")
        
        # Save model
        model_path = os.path.join(args.output_dir, f'lr_candle_{pos+1}.joblib')
        joblib.dump({'model': result['model'], 'scaler': result['scaler']}, model_path)
    
    # Summary
    print("-"*60)
    avg_acc = np.mean([r['accuracy'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    print(f"Average:  Acc={avg_acc:.3f}  F1={avg_f1:.3f}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Logistic Regression Results - {datetime.now()}\n")
        f.write(f"Regularization: {args.penalty.upper()}, C={args.C}\n")
        f.write(f"Features: {X_train_flat.shape[1]}\n\n")
        for pos, r in enumerate(results):
            f.write(f"Candle {pos+1}: Acc={r['accuracy']:.4f} F1={r['f1']:.4f}\n")
        f.write(f"\nAverage: Acc={avg_acc:.4f} F1={avg_f1:.4f}\n")
    
    print(f"\nModels saved to {args.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()