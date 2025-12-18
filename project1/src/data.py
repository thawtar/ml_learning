"""Data loading utilities."""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class TabularDataset(Dataset):
    """Generic dataset for tabular/array data (e.g., CFD simulation outputs)."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(X, y, cfg: dict):
    """Create train/val dataloaders from config."""
    dataset = TabularDataset(X, y)
    val_size = int(len(dataset) * cfg["data"]["val_split"])
    train_size = len(dataset) - val_size
    
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["project"]["seed"])
    )
    
    return (
        DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True),
        DataLoader(val_ds, batch_size=cfg["data"]["batch_size"])
    )
