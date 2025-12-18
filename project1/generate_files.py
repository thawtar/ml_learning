#!/usr/bin/env python3
"""
MLOps Pipeline Boilerplate Generator
Generates minimal, production-ready ML project structure.
"""

import os
from pathlib import Path

# ============================================================================
# FILE TEMPLATES
# ============================================================================

CONFIG_YAML = '''# MLOps Pipeline Configuration
project:
  name: "{project_name}"
  seed: 42

data:
  train_path: "data/train.csv"
  val_split: 0.2
  batch_size: 32

model:
  type: "mlp"
  hidden_dims: [128, 64]
  dropout: 0.1

training:
  epochs: 100
  lr: 1e-3
  early_stop_patience: 10

experiment:
  tracking_uri: "mlruns"  # MLflow tracking directory
  run_name: null          # Auto-generated if null
'''

MODEL_PY = '''"""Model definitions."""

import torch
import torch.nn as nn

class MLP(nn.Module):
    """Simple MLP for tabular data / CFD field predictions."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def build_model(cfg: dict, in_dim: int, out_dim: int) -> nn.Module:
    """Factory function to build model from config."""
    return MLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=cfg["model"]["hidden_dims"],
        dropout=cfg["model"]["dropout"]
    )
'''

DATA_PY = '''"""Data loading utilities."""

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
'''

TRAIN_PY = '''"""Training script with experiment tracking."""

import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime

import mlflow
from model import build_model
from data import get_dataloaders

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

def train(cfg_path: str = "config.yaml"):
    # Load config
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["project"]["seed"])
    
    # === REPLACE WITH YOUR DATA LOADING ===
    # Example: X, y = load_cfd_data(cfg["data"]["train_path"])
    X = np.random.randn(1000, 10)  # Placeholder
    y = np.random.randn(1000, 1)
    # ======================================
    
    train_dl, val_dl = get_dataloaders(X, y, cfg)
    in_dim, out_dim = X.shape[1], y.shape[1] if y.ndim > 1 else 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.MSELoss()
    
    # MLflow tracking
    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["project"]["name"])
    
    run_name = cfg["experiment"]["run_name"] or f"run_{datetime.now():%Y%m%d_%H%M%S}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "lr": cfg["training"]["lr"],
            "batch_size": cfg["data"]["batch_size"],
            "hidden_dims": str(cfg["model"]["hidden_dims"]),
        })
        
        best_val, patience = float("inf"), 0
        
        for epoch in range(cfg["training"]["epochs"]):
            # Train
            model.train()
            train_loss = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dl)
            
            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb), yb).item()
            val_loss /= len(val_dl)
            
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
            # Early stopping & checkpointing
            if val_loss < best_val:
                best_val, patience = val_loss, 0
                torch.save(model.state_dict(), "best_model.pt")
                mlflow.log_artifact("best_model.pt")
            else:
                patience += 1
                if patience >= cfg["training"]["early_stop_patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        mlflow.log_artifact(cfg_path)
    
    print(f"Training complete. Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)
'''

PREDICT_PY = '''"""Inference script."""

import yaml
import torch
import numpy as np
from model import build_model

def load_model(cfg_path: str = "config.yaml", weights: str = "best_model.pt", in_dim: int = 10, out_dim: int = 1):
    cfg = yaml.safe_load(open(cfg_path))
    model = build_model(cfg, in_dim, out_dim)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model

def predict(model, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.FloatTensor(X)).numpy()

if __name__ == "__main__":
    # Example usage
    model = load_model()
    X_test = np.random.randn(5, 10)  # Replace with actual data
    preds = predict(model, X_test)
    print("Predictions:", preds)
'''

REQUIREMENTS_TXT = '''torch>=2.0
numpy
pyyaml
mlflow>=2.0
'''

# ============================================================================
# GENERATOR LOGIC
# ============================================================================

FILES = {
    "config.yaml": CONFIG_YAML,
    "model.py": MODEL_PY,
    "data.py": DATA_PY,
    "train.py": TRAIN_PY,
    "predict.py": PREDICT_PY,
    "requirements.txt": REQUIREMENTS_TXT,
}

def generate(project_name: str = "mlops_project", output_dir: str = "."):
    """Generate MLOps boilerplate files."""
    out = Path(output_dir) / project_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)  # Data directory
    
    for fname, content in FILES.items():
        fpath = out / fname
        content = content.format(project_name=project_name) if "{project_name}" in content else content
        fpath.write_text(content.strip() + "\n")
        print(f"  Created: {fpath}")
    
    print(f"\n✓ Project '{project_name}' generated at {out.resolve()}")
    print(f"\nNext steps:")
    print(f"  cd {project_name}")
    print(f"  pip install -r requirements.txt")
    print(f"  python train.py --config config.yaml")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate MLOps pipeline boilerplate")
    parser.add_argument("--name", default="mlops_project", help="Project name")
    parser.add_argument("--output", default=".", help="Output directory")
    args = parser.parse_args()
    
    print(f"Generating MLOps boilerplate...")
    generate(args.name, args.output)