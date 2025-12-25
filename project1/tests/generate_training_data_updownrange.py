"""
Forex Training Data Generator - Three-Class Label Version (UP/SIDEWAYS/DOWN)
===========================================================================
Generates training data for AI forex prediction models with three-class classification.

Features per sample:
- OHLC sequence (Heikin-Ashi) for sequence_length time steps
- MACD and RSI indicators at current time step
- Three-class label: UP (2), SIDEWAYS (1), or DOWN (0)

Label Logic:
- Label = 2 (UP) if close[current + window_size] > close[current] + sideways_threshold
- Label = 1 (SIDEWAYS) if |close[current + window_size] - close[current]| <= sideways_threshold
- Label = 0 (DOWN) if close[current + window_size] < close[current] - sideways_threshold
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ForexIndicators:
    """Technical indicators calculator for forex data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.o = df['Open']
        self.h = df['High']
        self.l = df['Low']
        self.c = df['Close']
        self.v = df.get('Vol', pd.Series(0, index=df.index))
    
    def rsi(self, period: int = 14) -> pd.Series:
        """Calculate RSI (0-1 normalized)."""
        delta = self.c.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return (100. - (100. / (1. + rs))) / 100.  # Normalized 0-1
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD components."""
        ema_fast = self.c.ewm(span=fast, adjust=False).mean()
        ema_slow = self.c.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            'macd': macd_line, 
            'signal': signal_line, 
            'histogram': histogram
        })
    
    def sma(self, period: int = 20) -> pd.Series:
        return self.c.rolling(period).mean()
    
    def ema(self, period: int = 20) -> pd.Series:
        return self.c.ewm(span=period, adjust=False).mean()
    
    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range for volatility."""
        tr = pd.concat([
            self.h - self.l,
            (self.h - self.c.shift()).abs(),
            (self.l - self.c.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()


class HeikinAshiDataGenerator:
    """
    Training data generator for forex prediction with three-class labels.
    
    Generates sequences with:
    - Heikin-Ashi OHLC data (sequence_length time steps)
    - Technical indicators (MACD, RSI) at current step
    - Three-class label: UP (2), SIDEWAYS (1), or DOWN (0)
    """
    
    def __init__(self, data_file: str, sequence_length: int = 9, 
                 prediction_horizon: int = 4, sideways_threshold: float = 0.001):
        """
        Initialize the data generator.
        
        Args:
            data_file: Path to CSV file with OHLC data
            sequence_length: Number of past candles for input sequence
            prediction_horizon: Number of steps ahead to check price
            sideways_threshold: Price change threshold to consider sideways (absolute value)
                              e.g., 0.001 = 0.1% price change = sideways
                              For USDJPY at 150.00, this would be ±0.15 (15 pips)
        """
        self.data_file = data_file
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.sideways_threshold = sideways_threshold
        
        self.df = None
        self.ha_df = None
        self.indicators_df = None
        
        logger.info(f"Initialized HeikinAshiDataGenerator (Three-Class Mode):")
        logger.info(f"  - Data file: {data_file}")
        logger.info(f"  - Sequence length: {sequence_length}")
        logger.info(f"  - Prediction horizon: {prediction_horizon} steps ahead")
        logger.info(f"  - Sideways threshold: {sideways_threshold} (±{sideways_threshold*100:.2f}%)")
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate forex data from CSV."""
        logger.info(f"Loading data from {self.data_file}")
        
        self.df = pd.read_csv(self.data_file)
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Clean data
        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.df)} candles")
        logger.info(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def calculate_heikin_ashi(self) -> pd.DataFrame:
        """Convert OHLC to Heikin-Ashi candles."""
        if self.df is None:
            raise ValueError("Load data first using load_data()")
        
        logger.info("Calculating Heikin-Ashi candles...")
        
        ha_data = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            
            # HA Close = (O + H + L + C) / 4
            ha_close = (row['Open'] + row['High'] + row['Low'] + row['Close']) / 4.0
            
            if i == 0:
                # First candle: HA Open = (O + C) / 2
                ha_open = (row['Open'] + row['Close']) / 2.0
            else:
                # Subsequent: HA Open = (prev_HA_Open + prev_HA_Close) / 2
                prev = ha_data[i-1]
                ha_open = (prev['HA_Open'] + prev['HA_Close']) / 2.0
            
            ha_high = max(row['High'], ha_open, ha_close)
            ha_low = min(row['Low'], ha_open, ha_close)
            
            ha_data.append({
                'HA_Open': ha_open,
                'HA_High': ha_high,
                'HA_Low': ha_low,
                'HA_Close': ha_close
            })
        
        self.ha_df = pd.DataFrame(ha_data)
        logger.info(f"Generated {len(self.ha_df)} Heikin-Ashi candles")
        
        return self.ha_df
    
    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate technical indicators on original OHLC data."""
        if self.df is None:
            raise ValueError("Load data first")
        
        logger.info("Calculating technical indicators...")
        
        indicators = ForexIndicators(self.df)
        
        # Calculate indicators
        rsi = indicators.rsi(14)
        macd_data = indicators.macd(12, 26, 9)
        atr = indicators.atr(14)
        
        # Normalize MACD components by ATR for scale invariance
        self.indicators_df = pd.DataFrame({
            'RSI': rsi,
            'MACD': macd_data['macd'] / (atr + 1e-8),
            'MACD_Signal': macd_data['signal'] / (atr + 1e-8),
            'MACD_Histogram': macd_data['histogram'] / (atr + 1e-8),
            'ATR': atr
        })
        
        # Count NaN values before dropping
        nan_counts = self.indicators_df.isnull().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            logger.warning(f"Found {total_nans} NaN values in indicators:")
            for col, count in nan_counts.items():
                if count > 0:
                    logger.warning(f"  - {col}: {count} NaN values")
        
        logger.info("Indicators calculated: RSI, MACD, MACD_Signal, MACD_Histogram, ATR")
        
        return self.indicators_df
    
    def _remove_nan_rows(self) -> None:
        """Remove rows with NaN values from both ha_df and indicators_df."""
        if self.ha_df is None:
            raise ValueError("Calculate Heikin-Ashi first")
        
        initial_length = len(self.ha_df)
        
        if self.indicators_df is not None:
            nan_mask = self.indicators_df.isnull().any(axis=1)
            valid_indices = ~nan_mask
            
            num_nan_rows = nan_mask.sum()
            
            if num_nan_rows > 0:
                logger.info(f"Removing {num_nan_rows} rows with NaN values from indicators...")
                
                self.ha_df = self.ha_df[valid_indices].reset_index(drop=True)
                self.indicators_df = self.indicators_df[valid_indices].reset_index(drop=True)
                
                if self.df is not None:
                    self.df = self.df[valid_indices].reset_index(drop=True)
                
                logger.info(f"  - Rows before: {initial_length}")
                logger.info(f"  - Rows after: {len(self.ha_df)}")
                logger.info(f"  - Rows removed: {num_nan_rows}")
            else:
                logger.info("No NaN values found in indicators")
        else:
            nan_mask = self.ha_df.isnull().any(axis=1)
            valid_indices = ~nan_mask
            
            num_nan_rows = nan_mask.sum()
            
            if num_nan_rows > 0:
                logger.info(f"Removing {num_nan_rows} rows with NaN values from Heikin-Ashi data...")
                self.ha_df = self.ha_df[valid_indices].reset_index(drop=True)
                
                if self.df is not None:
                    self.df = self.df[valid_indices].reset_index(drop=True)
                
                logger.info(f"  - Rows before: {initial_length}")
                logger.info(f"  - Rows after: {len(self.ha_df)}")
    
    def generate_sequences(self, include_indicators: bool = True,
                          standardize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate input sequences and labels for training.
        
        Args:
            include_indicators: Whether to include technical indicators
            standardize: Whether to standardize OHLC sequences
            
        Returns:
            X: Input sequences
            y: Three-class labels (0=DOWN, 1=SIDEWAYS, 2=UP)
        """
        if self.ha_df is None:
            self.calculate_heikin_ashi()
        if include_indicators and self.indicators_df is None:
            self.calculate_indicators()
        
        self._remove_nan_rows()
        
        logger.info("Generating training sequences...")
        
        # Verify no NaN values remain
        if self.ha_df.isnull().any().any():
            raise ValueError("NaN values still present in Heikin-Ashi data after cleaning!")
        
        if include_indicators and self.indicators_df.isnull().any().any():
            raise ValueError("NaN values still present in indicators after cleaning!")
        
        # Get data arrays
        ha_ohlc = self.ha_df[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']].values
        
        if include_indicators:
            indicators = self.indicators_df[['RSI', 'MACD', 'MACD_Signal']].values
        
        num_samples = len(ha_ohlc) - self.sequence_length - self.prediction_horizon
        
        if num_samples <= 0:
            raise ValueError(
                f"Insufficient data after removing NaN rows. "
                f"Available: {len(ha_ohlc)}, "
                f"Required: {self.sequence_length + self.prediction_horizon + 1}"
            )
        
        sequences = []
        
        for i in range(num_samples):
            seq_start = i
            seq_end = i + self.sequence_length
            ohlc_seq = ha_ohlc[seq_start:seq_end].copy()
            ohlc_seq = ohlc_seq.astype(np.float32)
            ohlc_seq = np.array(ohlc_seq)
            ohlc_seq = np.log10(ohlc_seq + 1e-8)
            
            if np.isnan(ohlc_seq).any():
                logger.warning(f"NaN detected in sequence {i}, skipping...")
                continue
            
            if standardize:
                mean = ohlc_seq.mean()
                std = ohlc_seq.std()
                if std < 1e-8:
                    std = 1e-8
                ohlc_seq = (ohlc_seq - mean) / std
            
            if include_indicators:
                current_idx = seq_end - 1
                ind_vals = indicators[current_idx]
                
                if np.isnan(ind_vals).any():
                    logger.warning(f"NaN detected in indicators at index {current_idx}, skipping...")
                    continue
                
                indicator_row = np.concatenate([ind_vals, [0.0]]).reshape(1, 4)
                full_sequence = np.vstack([ohlc_seq, indicator_row])
                
                sequences.append(full_sequence)
            else:
                sequences.append(ohlc_seq)
        
        X = np.array(sequences, dtype=np.float32)
        
        if include_indicators:
            logger.info(f"  Features: OHLC sequence + indicators as additional row")
            logger.info(f"  OHLC timesteps: {self.sequence_length}, Indicator timestep: 1")
            logger.info(f"  Total sequence length: {self.sequence_length + 1}")
        else:
            logger.info(f"  Features: 4 OHLC per timestep")
        
        # Generate labels
        y = self.generate_labels()
        
        # Ensure X and y have the same length
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]
        
        # Final NaN check
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            raise ValueError(f"Final check failed: {nan_count} NaN values found in X!")
        
        if np.isnan(y).any():
            nan_count = np.isnan(y).sum()
            raise ValueError(f"Final check failed: {nan_count} NaN values found in y!")
        
        logger.info(f"Generated {len(X)} samples (after NaN removal)")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape} - three-class labels (0=DOWN, 1=SIDEWAYS, 2=UP)")
        logger.info(f"  ✓ No NaN values in final dataset")
        
        return X, y

    def generate_labels(self) -> np.ndarray:
        """
        Generate three-class labels for each sample.
        
        Label Logic:
        - Check price at position: current_index + prediction_horizon
        - price_change = HA_Close[current + horizon] - HA_Close[current]
        - If price_change > +sideways_threshold → Label = 2 (UP)
        - If |price_change| <= sideways_threshold → Label = 1 (SIDEWAYS)
        - If price_change < -sideways_threshold → Label = 0 (DOWN)
        
        Returns:
            Array of shape (num_samples,) with three-class labels
        """
        if self.ha_df is None:
            raise ValueError("Calculate Heikin-Ashi first")
        
        logger.info(f"Generating three-class labels (checking price {self.prediction_horizon} steps ahead)...")
        logger.info(f"  Sideways threshold: ±{self.sideways_threshold} (±{self.sideways_threshold*100:.2f}%)")
        
        ha_close = self.ha_df['HA_Close'].values
        
        if np.isnan(ha_close).any():
            raise ValueError("NaN values found in HA_Close! Run _remove_nan_rows() first.")
        
        num_samples = len(ha_close) - self.sequence_length - self.prediction_horizon
        
        if num_samples <= 0:
            raise ValueError(
                f"Insufficient data for label generation. "
                f"Available: {len(ha_close)}, "
                f"Required: {self.sequence_length + self.prediction_horizon + 1}"
            )
        
        labels = []
        
        for i in range(num_samples):
            current_idx = i + self.sequence_length - 1
            current_close = ha_close[current_idx]
            
            future_idx = current_idx + self.prediction_horizon
            future_close = ha_close[future_idx]
            
            # Calculate price change (absolute)
            price_change = future_close - current_close
            
            # Three-class classification
            if price_change > self.sideways_threshold:
                label = 2  # UP / BULLISH
            elif price_change < -self.sideways_threshold:
                label = 0  # DOWN / BEARISH
            else:
                label = 1  # SIDEWAYS / NEUTRAL
            
            labels.append(label)
        
        labels_array = np.array(labels, dtype=np.float32)
        
        # Log label distribution
        down_count = (labels_array == 0).sum()
        sideways_count = (labels_array == 1).sum()
        up_count = (labels_array == 2).sum()
        total_count = len(labels_array)
        
        logger.info(f"Label Distribution:")
        logger.info(f"  - Total samples: {total_count}")
        logger.info(f"  - DOWN (0):     {int(down_count)} ({down_count/total_count*100:.1f}%)")
        logger.info(f"  - SIDEWAYS (1): {int(sideways_count)} ({sideways_count/total_count*100:.1f}%)")
        logger.info(f"  - UP (2):       {int(up_count)} ({up_count/total_count*100:.1f}%)")
        
        return labels_array
    
    def save_training_data(self, output_dir: str = 'data',
                          include_indicators: bool = True) -> Dict[str, str]:
        """Save generated training data to numpy files."""
        os.makedirs(output_dir, exist_ok=True)
        
        X, y = self.generate_sequences(include_indicators=include_indicators)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_seq{self.sequence_length}_horizon{self.prediction_horizon}_three_class.npz"
        filepath = os.path.join(output_dir, filename)
        
        np.savez(filepath, X=X, y=y)
        
        logger.info(f"Saved training data to {filepath}")
        
        return {'data_file': filepath, 'X_shape': X.shape, 'y_shape': y.shape}
    
    def save_training_data_with_splits(self, output_dir: str = 'data_splits',
                                       sequence_length: int = None,
                                       train_ratio: float = 0.6,
                                       val_ratio: float = 0.2,
                                       test_ratio: float = 0.2,
                                       include_indicators: bool = False) -> Dict:
        """Generate and save train/val/test splits."""
        if sequence_length:
            self.sequence_length = sequence_length
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all data
        X, y = self.generate_sequences(include_indicators=include_indicators)
        
        n_samples = len(X)
        
        # Calculate split indices (chronological, no shuffle)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split data
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Save splits
        train_file = os.path.join(output_dir, 'train_data.npz')
        val_file = os.path.join(output_dir, 'val_data.npz')
        test_file = os.path.join(output_dir, 'test_data.npz')
        
        np.savez(train_file, X=X_train, y=y_train)
        np.savez(val_file, X=X_val, y=y_val)
        np.savez(test_file, X=X_test, y=y_test)
        
        # Log split information
        logger.info(f"\n{'='*60}")
        logger.info(f"Data splits saved to {output_dir}:")
        logger.info(f"  Train: {len(X_train)} samples ({train_ratio*100:.0f}%)")
        logger.info(f"  Val:   {len(X_val)} samples ({val_ratio*100:.0f}%)")
        logger.info(f"  Test:  {len(X_test)} samples ({test_ratio*100:.0f}%)")
        
        # Label distribution per split
        logger.info(f"\nLabel Distribution (DOWN/SIDEWAYS/UP):")
        for name, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            down_pct = (labels == 0).sum() / len(labels) * 100
            sideways_pct = (labels == 1).sum() / len(labels) * 100
            up_pct = (labels == 2).sum() / len(labels) * 100
            logger.info(f"  {name:5s}: DOWN {down_pct:.1f}% | SIDEWAYS {sideways_pct:.1f}% | UP {up_pct:.1f}%")
        
        logger.info(f"{'='*60}\n")
        
        return {
            'train_file': train_file,
            'val_file': val_file,
            'test_file': test_file,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_dim': X.shape[2],
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'label_type': 'three_class',
            'num_classes': 3
        }
    
    def save_features_to_csv(self, output_file: str = 'features_raw.csv',
                            include_indicators: bool = True,
                            max_samples: int = None) -> str:
        """Save raw features to CSV for visualization."""
        logger.info("Generating raw features for CSV export...")
        
        X, y = self.generate_sequences(
            include_indicators=include_indicators,
            standardize=True
        )
        
        if max_samples is not None and max_samples < len(X):
            X = X[:max_samples]
            y = y[:max_samples]
            logger.info(f"Limited to {max_samples} samples for CSV export")
        
        num_samples, seq_len, num_features = X.shape
        
        columns = []
        
        if include_indicators:
            for t in range(seq_len - 1):
                for feat_name in ['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']:
                    columns.append(f'timestep_{t}_{feat_name}')
            
            columns.append('RSI')
            columns.append('MACD')
            columns.append('MACD_Signal')
            columns.append('indicator_padding')
        else:
            for t in range(seq_len):
                for feat_name in ['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']:
                    columns.append(f'timestep_{t}_{feat_name}')
        
        columns.append('label')
        
        X_flat = X.reshape(num_samples, -1)
        
        df = pd.DataFrame(X_flat, columns=columns[:-1])
        df['label'] = y
        
        df.insert(0, 'sample_id', range(len(df)))
        
        df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(df)} samples to {output_file}")
        logger.info(f"  - Label distribution: DOWN={int((y==0).sum())}, SIDEWAYS={int((y==1).sum())}, UP={int((y==2).sum())}")
        
        return output_file


def main():
    """Example usage of the data generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate forex training data (Three-Class: UP/SIDEWAYS/DOWN)')
    parser.add_argument('--data_file', type=str, default='USDJPY_H4.csv',
                       help='Input CSV file with OHLC data')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Number of past candles for input')
    parser.add_argument('--prediction_horizon', type=int, default=10,
                       help='Number of steps ahead to predict')
    parser.add_argument('--sideways_threshold', type=float, default=0.3,
                       help='Threshold for sideways classification (e.g., 0.001 = 0.1%% price change)')
    parser.add_argument('--output_dir', type=str, default='data_splits_three_class',
                       help='Output directory for training data')
    parser.add_argument('--include_indicators', action='store_true',
                       help='Include MACD/RSI in features', default=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--save_csv', action='store_true',
                       help='Save raw features to CSV for visualization')
    parser.add_argument('--csv_output', type=str, default='features_raw.csv',
                       help='Output CSV filename for raw features')
    parser.add_argument('--csv_max_samples', type=int, default=1000,
                       help='Maximum samples to export to CSV')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FOREX TRAINING DATA GENERATOR - THREE-CLASS MODE")
    print("="*60)
    print(f"Label Strategy: Three-class classification")
    print(f"  - UP (2)       if price_change > +{args.sideways_threshold}")
    print(f"  - SIDEWAYS (1) if |price_change| <= {args.sideways_threshold}")
    print(f"  - DOWN (0)     if price_change < -{args.sideways_threshold}")
    print("="*60)
    
    # Initialize generator
    generator = HeikinAshiDataGenerator(
        data_file=args.data_file,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        sideways_threshold=args.sideways_threshold
    )
    
    # Load and process data
    generator.load_data()
    generator.calculate_heikin_ashi()
    if args.include_indicators:
        generator.calculate_indicators()

    # Export CSV if requested
    if args.save_csv:
        print("\n" + "="*60)
        print("EXPORTING RAW FEATURES TO CSV")
        print("="*60)
        csv_path = generator.save_features_to_csv(
            output_file=args.csv_output,
            include_indicators=args.include_indicators,
            max_samples=args.csv_max_samples
        )
        print(f"Raw features saved to: {csv_path}")
    
    # Generate and save splits
    result = generator.save_training_data_with_splits(
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        include_indicators=args.include_indicators
    )
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Label Type: {result['label_type']}")
    print(f"Number of classes: {result['num_classes']}")
    print(f"Sequence length: {result['sequence_length']}")
    print(f"Prediction horizon: {result['prediction_horizon']} steps ahead")
    print(f"Sideways threshold: ±{args.sideways_threshold}")
    print(f"Feature dimension: {result['feature_dim']}")
    print(f"Train samples: {result['train_samples']}")
    print(f"Val samples: {result['val_samples']}")
    print(f"Test samples: {result['test_samples']}")
    print(f"\nFiles saved to: {args.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    main()