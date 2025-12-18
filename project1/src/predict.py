"""Inference script."""

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
