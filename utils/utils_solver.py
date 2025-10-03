import csv, datetime as dt
import torch 
from typing import Dict, Any, List, Tuple
import numpy as np
import torch.nn as nn
import os

def now_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def set_seeds(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_kv_csv(path: str, kv: Dict[str, Any]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["key", "value"])
        for k, v in kv.items(): w.writerow([k, v])

def write_curves_csv(path: str, rows: List[Dict[str, Any]]):
    fieldnames = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in fieldnames})

def write_metrics_csv(path: str, rows: List[Dict[str, Any]]):
    fieldnames = ["split", "loss", "acc", "f1_macro"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fieldnames})

def write_class_report_csv(path: str, report_dict: Dict[str, Dict[str, float]]):
    fieldnames = ["label", "precision", "recall", "f1-score", "support"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for label, stats in report_dict.items():
            if not isinstance(stats, dict):  # ignora 'accuracy' scalar
                continue
            w.writerow({
                "label": label,
                "precision": stats.get("precision", ""),
                "recall": stats.get("recall", ""),
                "f1-score": stats.get("f1-score", ""),
                "support": stats.get("support", "")
            })

def write_confusion_csv(path: str, cm: np.ndarray):
    n = cm.shape[0]
    headers = ["true\\pred"] + [f"pred_{j}" for j in range(n)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(headers)
        for i in range(n): w.writerow([f"true_{i}"] + list(map(int, cm[i])))

def write_predictions_csv(path: str, y_true: np.ndarray, y_pred: np.ndarray):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["idx", "y_true", "y_pred"])
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            w.writerow([i, int(t), int(p)])

def evaluate_loss_acc(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, total_count, correct = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb); loss = crit(logits, yb)
            total_loss += loss.item() * xb.size(0); total_count += xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
    return total_loss / total_count, correct / total_count

def predict_all(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []; model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); logits = model(xb)
            ys.append(yb.numpy()); ps.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

class EarlyStopping:
    def __init__(self, mode: str = "min", patience: int = 20):
        assert mode in ("min", "max")
        self.mode = mode; self.patience = patience
        self.best = None; self.count = 0; self.best_epoch = -1
        self.is_better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
    def step(self, metric_value: float, epoch: int) -> bool:
        if self.best is None or self.is_better(metric_value, self.best):
            self.best, self.best_epoch, self.count = metric_value, epoch, 0
            return False
        self.count += 1
        return self.count > self.patience
