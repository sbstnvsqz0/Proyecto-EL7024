from typing import List
import torch
import torch.nn as nn

def activation_fn(name: str):
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "silu": return nn.SiLU()
    if name == "tanh": return nn.Tanh()
    raise ValueError(f"Activación desconocida: {name}")

class GaussianDropout(nn.Module):
    def __init__(self, p: float = 0.2):
        """
        p: dropout rate (0 <= p < 1)
        """
        super(GaussianDropout, self).__init__()
        if p < 0 or p >= 1:
            raise ValueError("Dropout probability must be in [0, 1).")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        std = (self.p / (1 - self.p)) ** 0.5
        noise = torch.randn_like(x) * std + 1.0
        return x * noise

class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, activation: str = "relu"):
        super().__init__()
        layers = []
        prev = in_dim
        act = activation_fn(activation)
        for h in hidden:
            layers += [nn.Linear(prev, h), act.__class__()] 
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, activation: str = "relu",dropout: str = "wout", dropout_prob: float = 0.2):
        super().__init__()
        assert dropout in ["wout","vanilla","gaussian"], "Dropout puede tomar solo los valores wout, vanilla, gaussian"
        layers = []
        prev = in_dim
        act = activation_fn(activation)
        dropout = None if dropout=="wout" else nn.Dropout() if dropout == "vanilla" else GaussianDropout()
        for h in hidden:
            layers += [nn.Linear(prev, h), act.__class__()] 
            layers += [dropout.__class__(dropout_prob)] if dropout is not None else []
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
