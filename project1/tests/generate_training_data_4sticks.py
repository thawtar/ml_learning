"""
Forex Training Data Generator
=============================
Generates training data for AI forex prediction models.

Features per sample:
- OHLC sequence (Heikin-Ashi) for sequence_length time steps
- MACD and RSI indicators at current time step
- Labels: Binary price action for window_size future candles
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
    Complete training data generator for forex prediction.
    
    Generates sequences with:
    - Heikin-Ashi OHLC data (sequence_length time steps)
    - Technical indicators (MACD, RSI) at current step
    - Binary labels for future price action (window_size candles)
    """
    
    def __init__(self, data_file: str, sequence_length: int = 9, 
                 future_window: int = 4, pip_threshold: float = 0.0):
        """
        Initialize the data generator.
        
        Args:
            data_file: Path to CSV file with OHLC data
            sequence_length: Number of past candles for input sequence
            future_window: Number of future candles to predict
            pip_threshold: Minimum price change to consider bullish (default 0)
        """
        self.data_file = data_file
        self.sequence_length = sequence_length
        self.future_window = future_window
        self.pip_threshold = pip_threshold
        
        self.df = None
        self.ha_df = None
        self.indicators_df = None
        
        logger.info(f"Initialized HeikinAshiDataGenerator:")
        logger.info(f"  - Data file: {data_file}")
        logger.info(f"  - Sequence length: {sequence_length}")
        logger.info(f"  - Future window: {future_window}")
    
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
            'MACD': macd_data['macd'] / (atr + 1e-8),  # Normalized
            'MACD_Signal': macd_data['signal'] / (atr + 1e-8),
            'MACD_Histogram': macd_data['histogram'] / (atr + 1e-8),
            'ATR': atr
        })
        
        logger.info("Indicators calculated: RSI, MACD, MACD_Signal, MACD_Histogram, ATR")
        
        return self.indicators_df
    
    def generate_labels(self) -> np.ndarray:
        """
        Generate binary labels for future price action.
        
        For each position, label is 1 if HA_Close[t+i] > HA_Close[t], else 0.
        Returns array of shape (num_samples, future_window)
        """
        if self.ha_df is None:
            raise ValueError("Calculate Heikin-Ashi first")
        
        logger.info(f"Generating labels for {self.future_window} future candles...")
        
        ha_close = self.ha_df['HA_Close'].values
        num_samples = len(ha_close) - self.sequence_length - self.future_window
        
        labels = []
        for i in range(num_samples):
            current_idx = i + self.sequence_length - 1
            current_close = ha_close[current_idx]
            
            # Binary label for each future candle
            future_labels = []
            for j in range(1, self.future_window + 1):
                future_close = ha_close[current_idx + j]
                # 1 if bullish (price went up), 0 if bearish
                label = 1 if future_close > current_close + self.pip_threshold else 0
                future_labels.append(label)
            
            labels.append(future_labels)
        
        labels_array = np.array(labels, dtype=np.float32)
        
        # Log label distribution
        for i in range(self.future_window):
            bullish_pct = labels_array[:, i].mean() * 100
            logger.info(f"  Candle {i+1}: {bullish_pct:.1f}% bullish")
        
        return labels_array
    
    def generate_sequences(self, include_indicators: bool = True,
                          standardize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
       
        if self.ha_df is None:
            self.calculate_heikin_ashi()
        if include_indicators and self.indicators_df is None:
            self.calculate_indicators()
        
        logger.info("Generating training sequences...")
        
        # Get data arrays
        ha_ohlc = self.ha_df[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']].values
        
        if include_indicators:
            # Use RSI, MACD, MACD_Signal at each timestep
            indicators = self.indicators_df[['RSI', 'MACD', 'MACD_Signal']].values
        
        num_samples = len(ha_ohlc) - self.sequence_length - self.future_window
        
        sequences = []
        indicator_features = []
        
        for i in range(num_samples):
            # Extract OHLC sequence
            seq_start = i
            seq_end = i + self.sequence_length
            ohlc_seq = ha_ohlc[seq_start:seq_end].copy()
            
            if standardize:
                # Standardize OHLC sequence (zero mean, unit variance)
                mean = ohlc_seq.mean()
                std = ohlc_seq.std()
                if std < 1e-8:
                    std = 1e-8
                ohlc_seq = (ohlc_seq - mean) / std
            
            sequences.append(ohlc_seq)
            
            if include_indicators:
                # Get indicators at current time step (end of sequence)
                current_idx = seq_end - 1
                ind_vals = indicators[current_idx]
                indicator_features.append(ind_vals)
        
        X_ohlc = np.array(sequences, dtype=np.float32)
        
        if include_indicators:
            # Append indicators as additional features at each timestep
            # Broadcast indicator values across sequence
            X_indicators = np.array(indicator_features, dtype=np.float32)
            # Expand to (samples, sequence_length, num_indicators)
            X_indicators_expanded = np.tile(
                X_indicators[:, np.newaxis, :], 
                (1, self.sequence_length, 1)
            )
            X = np.concatenate([X_ohlc, X_indicators_expanded], axis=2)
            logger.info(f"  Features: 4 OHLC + 3 indicators = 7 per timestep")
        else:
            X = X_ohlc
            logger.info(f"  Features: 4 OHLC per timestep")
        
        # Generate labels
        y = self.generate_labels()
        
        logger.info(f"Generated {len(X)} samples")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        
        return X, y
    
    def save_training_data(self, output_dir: str = 'data',
                          include_indicators: bool = True) -> Dict[str, str]:
        """Save generated training data to numpy files."""
        os.makedirs(output_dir, exist_ok=True)
        
        X, y = self.generate_sequences(include_indicators=include_indicators)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_seq{self.sequence_length}_fut{self.future_window}.npz"
        filepath = os.path.join(output_dir, filename)
        
        np.savez(filepath, X=X, y=y)
        
        logger.info(f"Saved training data to {filepath}")
        
        return {'data_file': filepath, 'X_shape': X.shape, 'y_shape': y.shape}
    
    def save_training_data_with_splits(self, output_dir: str = 'data_splits',
                                       sequence_length: int = 10,
                                       train_ratio: float = 0.6,
                                       val_ratio: float = 0.2,
                                       test_ratio: float = 0.2,
                                       include_indicators: bool = False) -> Dict:
        
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
        logger.info(f"\nData splits saved to {output_dir}:")
        logger.info(f"  Train: {len(X_train)} samples ({train_ratio*100:.0f}%)")
        logger.info(f"  Val:   {len(X_val)} samples ({val_ratio*100:.0f}%)")
        logger.info(f"  Test:  {len(X_test)} samples ({test_ratio*100:.0f}%)")
        
        # Label distribution per split
        for name, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            bullish_pct = labels.mean() * 100
            logger.info(f"  {name} bullish rate: {bullish_pct:.1f}%")
        
        return {
            'train_file': train_file,
            'val_file': val_file,
            'test_file': test_file,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'feature_dim': X.shape[2],
            'sequence_length': self.sequence_length,
            'future_window': self.future_window
        }


def main():
    """Example usage of the data generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate forex training data')
    parser.add_argument('--data_file', type=str, default='USDJPY_H4.csv',
                       help='Input CSV file with OHLC data')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Number of past candles for input')
    parser.add_argument('--future_window', type=int, default=4,
                       help='Number of future candles to predict')
    parser.add_argument('--output_dir', type=str, default='data_splits',
                       help='Output directory for training data')
    parser.add_argument('--include_indicators', action='store_true',
                       help='Include MACD/RSI in features')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    
    print("="*60)
    print("FOREX TRAINING DATA GENERATOR")
    print("="*60)
    
    # Initialize generator
    generator = HeikinAshiDataGenerator(
        data_file=args.data_file,
        sequence_length=args.sequence_length,
        future_window=args.future_window
    )
    
    # Load and process data
    generator.load_data()
    generator.calculate_heikin_ashi()
    generator.calculate_indicators()
    
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
    print(f"Sequence length: {result['sequence_length']}")
    print(f"Future window: {result['future_window']}")
    print(f"Feature dimension: {result['feature_dim']}")
    print(f"Train samples: {result['train_samples']}")
    print(f"Val samples: {result['val_samples']}")
    print(f"Test samples: {result['test_samples']}")
    print(f"\nFiles saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()