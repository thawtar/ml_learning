"""Training script with experiment tracking."""

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
