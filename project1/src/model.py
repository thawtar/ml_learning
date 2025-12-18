"""Model definitions."""

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
