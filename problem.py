# problem.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Literal, Tuple
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_blobs

import torch
from torch.utils.data import TensorDataset, DataLoader

TaskType = Literal["classification", "regression"]

@dataclass
class ProblemSpec:
    name: str
    task: TaskType
    n_features: int
    n_classes: int | None
    build_datasets: Callable[[Dict[str, Any]], Dict[str, Any]]
    default_model_kwargs: Dict[str, Any]
    default_train_kwargs: Dict[str, Any]
    plotting: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["build_datasets"] = f"<callable:{self.build_datasets.__name__}>"
        return d


def _standardize_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float,
    test_size: float,
    stratify: bool,
    seed: int
) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    """Split (train/val/test) and standardize using train statistics only."""
    if stratify:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=val_size + test_size, random_state=seed, stratify=y
        )
        rel_val = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.0
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=(1 - rel_val), random_state=seed, stratify=y_tmp
        )
    else:
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=val_size + test_size, random_state=seed
        )
        rel_val = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.0
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=(1 - rel_val), random_state=seed
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    splits = {
        "train": (X_train.astype(np.float32), y_train.astype(np.int64)),
        "val":   (X_val.astype(np.float32),   y_val.astype(np.int64)),
        "test":  (X_test.astype(np.float32),  y_test.astype(np.int64)),
        "all":   (np.vstack([X_train, X_val, X_test]).astype(np.float32),
                  np.concatenate([y_train, y_val, y_test]).astype(np.int64))
    }
    return splits, scaler


def _to_loaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    shuffle_train: bool = True
) -> Dict[str, DataLoader]:
    def to_tensor_dataset(X, y):
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        return TensorDataset(X_t, y_t)

    loaders = {
        "train": DataLoader(to_tensor_dataset(*splits["train"]), batch_size=batch_size, shuffle=shuffle_train),
        "val":   DataLoader(to_tensor_dataset(*splits["val"]),   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(to_tensor_dataset(*splits["test"]),  batch_size=batch_size, shuffle=False),
    }
    return loaders


# --------- Problem builders ---------

def build_moons(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    cfg keys (con defaults razonables):
      n_samples:int=4000, noise:float=0.2, seed:int=0,
      val_size:float=0.2, test_size:float=0.2,
      batch_size:int=128, stratify:bool=True
    """
    n_samples = int(cfg.get("n_samples", 4000))
    noise     = float(cfg.get("noise", 0.2))
    seed      = int(cfg.get("seed", 0))
    val_size  = float(cfg.get("val_size", 0.2))
    test_size = float(cfg.get("test_size", 0.2))
    batch_sz  = int(cfg.get("batch_size", 128))
    stratify  = bool(cfg.get("stratify", True))

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    splits, scaler = _standardize_split(X, y, val_size, test_size, stratify, seed)
    loaders = _to_loaders(splits, batch_sz, shuffle_train=True)

    return {
        "loaders": loaders,
        "n_features": 2,
        "n_classes": 2,
        "scaler": scaler,
        "X_vis": splits["all"][0],   # para decision boundary
        "y_vis": splits["all"][1],
        "class_names": ["class0", "class1"],
        "plotting": {"decision_boundary_2d": True}
    }


def build_blobs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    cfg keys:
      n_samples:int=6000, centers:int=4, cluster_std:float=1.5, seed:int=0,
      val_size:float=0.2, test_size:float=0.2,
      batch_size:int=128, stratify:bool=True
    """
    n_samples  = int(cfg.get("n_samples", 6000))
    centers    = int(cfg.get("centers", 4))
    cluster_sd = float(cfg.get("cluster_std", 1.5))
    seed       = int(cfg.get("seed", 0))
    val_size   = float(cfg.get("val_size", 0.2))
    test_size  = float(cfg.get("test_size", 0.2))
    batch_sz   = int(cfg.get("batch_size", 128))
    stratify   = bool(cfg.get("stratify", True))

    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_sd, random_state=seed)
    splits, scaler = _standardize_split(X, y, val_size, test_size, stratify, seed)
    loaders = _to_loaders(splits, batch_sz, shuffle_train=True)

    return {
        "loaders": loaders,
        "n_features": X.shape[1],
        "n_classes": centers,
        "scaler": scaler,
        "X_vis": splits["all"][0][:, :2] if X.shape[1] >= 2 else splits["all"][0],  # para 2D si aplica
        "y_vis": splits["all"][1],
        "class_names": [f"class{i}" for i in range(centers)],
        "plotting": {"decision_boundary_2d": (X.shape[1] == 2)}
    }

# Registro
PROBLEMS: Dict[str, ProblemSpec] = {}

PROBLEMS["moons"] = ProblemSpec(
    name="moons",
    task="classification",
    n_features=2,
    n_classes=2,
    build_datasets=build_moons,
    default_model_kwargs={"hidden_sizes": [64, 64], "activation": "relu"},
    default_train_kwargs={"epochs": 200, "batch_size": 128, "lr": 1e-3, "weight_decay": 0.0, "optimizer": "adam", "patience": 20, "val_metric": "val_loss"},
    plotting={"decision_boundary_2d": True}
)

PROBLEMS["blobs"] = ProblemSpec(
    name="blobs",
    task="classification",
    n_features=2,    # nota: si cambias centers o dims puedes sobreescribir en build_datasets
    n_classes=None,  # se fija en runtime
    build_datasets=build_blobs,
    default_model_kwargs={"hidden_sizes": [64, 64], "activation": "relu"},
    default_train_kwargs={"epochs": 200, "batch_size": 128, "lr": 1e-3, "weight_decay": 0.0, "optimizer": "adam", "patience": 20, "val_metric": "val_loss"},
    plotting={"decision_boundary_2d": True}
)

def get_problem(name: str) -> ProblemSpec:
    if name not in PROBLEMS:
        raise ValueError(f"Problem '{name}' no registrado. Disponibles: {list(PROBLEMS.keys())}")
    return PROBLEMS[name]
