# solve_simple.py
from __future__ import annotations
import os, csv, datetime as dt
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from problem import get_problem, ProblemSpec

# =========================
# CONFIGURACIÓN EN CÓDIGO
# =========================
CONFIG: Dict[str, Any] = {
    "run": {
        "problem": "moons",                  # "moons" | "blobs"
        "outdir": "results",
        "method_dir": "dropout_vanilla_0_2",
        "seeds": [0, 1, 2],
        "save_model": False                  # mantenlo en False para solo-CSV
    },
    "data": {
        # Moons
        "n_samples": 4000,
        "noise": 0.2,
        # Blobs (ignorado si moons)
        "centers": 4,
        "cluster_std": 1.5,
        # Splits y batch
        "val_size": 0.2,
        "test_size": 0.2,
        "batch_size": 128,
        "stratify": True
    },
    "model": {
        "hidden_sizes": [64, 64],
        "activation": "relu",
        "dropout": {
            "type":"vanilla",
            "dropout_prob":0.2
        }
    },
    "train": {
        "epochs": 200,
        "optimizer": "adam",         # "adam" | "sgd"
        "lr": 1e-3,
        "weight_decay": 0.0,
        "patience": 20,              # early stopping
        "val_metric": "val_loss"     # "val_loss" (min) | "val_acc" (max)
    }
}

# =============== utilidades ===============

def now_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def activation_fn(name: str):
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "silu": return nn.SiLU()
    if name == "tanh": return nn.Tanh()
    raise ValueError(f"Activación desconocida: {name}")

#Otro tipo de dropout: implementacion https://discuss.pytorch.org/t/gaussiandropout-implementation/151756/4
class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p debe cumplir que 0 < p < 1")
        self.p = p
        
    def forward(self, x):
        if self.training:
            stddev = (self.p / (1.0 - self.p))**0.5
            epsilon = torch.randn_like(x) * stddev
            return x * epsilon
        else:
            return x

class DropoutMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, activation: str = "relu", dropout_type: str = "vanilla", dropout_prob: float = 0.2):
        assert dropout_type in ["vanilla","gaussian"], "Parametro dropout_type solo puede tomar valores vanilla o gaussian"
        super().__init__()
        layers = []
        prev = in_dim
        act = activation_fn(activation)
        dropout_fn = nn.Dropout() if dropout_type == "vanilla" else GaussianDropout()
        for h in hidden:
            layers += [nn.Linear(prev, h), act.__class__()]
            layers += [dropout_fn.__class__(dropout_prob)]  #Se añade dropout luego de activación en cada capa
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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

# =============== loop principal ===============

def train_one_seed(seed: int, cfg: Dict[str, Any], run_root: str) -> Dict[str, Any]:
    set_seeds(seed); device = device_auto()

    # Datos
    spec: ProblemSpec = get_problem(cfg["run"]["problem"])
    data_cfg = dict(cfg["data"]); data_cfg["seed"] = seed
    built = spec.build_datasets(data_cfg)
    loaders = built["loaders"]
    n_features, n_classes = built["n_features"], built["n_classes"]

    # Modelo
    model = DropoutMLP(n_features, cfg["model"]["hidden_sizes"], n_classes, cfg["model"]["activation"],cfg["model"]["dropout"]["type"],cfg["model"]["dropout"]["dropout_prob"]).to(device)

    # Optimizador
    if cfg["train"]["optimizer"].lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    elif cfg["train"]["optimizer"].lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"], momentum=0.9)
    else:
        raise ValueError("optimizer desconocido")
    crit = nn.CrossEntropyLoss()
    val_metric_name = cfg["train"]["val_metric"]
    stopper = EarlyStopping(mode=("min" if val_metric_name == "val_loss" else "max"), patience=cfg["train"]["patience"])

    # carpetas
    seed_root = os.path.join(run_root, f"seed_{seed}"); ensure_dir(seed_root)

    # Guardar args como CSV
    flat_args = {
        "problem": cfg["run"]["problem"], "method": cfg["run"]["method_dir"], "seed": seed,
        "dropout_type": cfg["model"]["dropout"]["type"], "dropout_prob": cfg["model"]["dropout"]["dropout_prob"],
        "hidden_sizes": "-".join(map(str, cfg["model"]["hidden_sizes"])), "activation": cfg["model"]["activation"],
        "optimizer": cfg["train"]["optimizer"], "lr": cfg["train"]["lr"], "weight_decay": cfg["train"]["weight_decay"],
        "epochs": cfg["train"]["epochs"], "patience": cfg["train"]["patience"], "val_metric": cfg["train"]["val_metric"],
        "batch_size": cfg["data"]["batch_size"], "val_size": cfg["data"]["val_size"], "test_size": cfg["data"]["test_size"],
        "stratify": cfg["data"]["stratify"], "device": str(device),
        # también guardar parámetros específicos para reconstruir datos en benchmark
        "n_samples": cfg["data"].get("n_samples",""), "noise": cfg["data"].get("noise",""),
        "centers": cfg["data"].get("centers",""), "cluster_std": cfg["data"].get("cluster_std","")
    }
    write_kv_csv(os.path.join(seed_root, "args.csv"), flat_args)

    # Entrenamiento
    curves = []; best_state = None; best_val_metric = None
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train(); epoch_loss = 0.0; epoch_count = 0; correct = 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits = model(xb); loss = crit(logits, yb)
            loss.backward(); opt.step()
            epoch_loss += loss.item() * xb.size(0); epoch_count += xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
        train_loss = epoch_loss / epoch_count; train_acc = correct / epoch_count
        val_loss, val_acc = evaluate_loss_acc(model, loaders["val"], device)
        curves.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})

        current = val_loss if val_metric_name == "val_loss" else val_acc
        if best_val_metric is None or (val_metric_name == "val_loss" and current < best_val_metric) or (val_metric_name == "val_acc" and current > best_val_metric):
            best_val_metric = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopper.step(current, epoch): break

    if best_state is not None: model.load_state_dict(best_state)

    # Guardar curvas
    write_curves_csv(os.path.join(seed_root, "curves.csv"), curves)

    # Predicciones por split (para benchmark)
    train_y_true, train_y_pred = predict_all(model, loaders["train"], device)
    val_y_true,   val_y_pred   = predict_all(model, loaders["val"], device)
    test_y_true,  test_y_pred  = predict_all(model, loaders["test"], device)
    write_predictions_csv(os.path.join(seed_root, "predictions_train.csv"), train_y_true, train_y_pred)
    write_predictions_csv(os.path.join(seed_root, "predictions_val.csv"),   val_y_true,   val_y_pred)
    write_predictions_csv(os.path.join(seed_root, "predictions_test.csv"),  test_y_true,  test_y_pred)

    # Métricas finales por split
    def loss_on(loader): l, _ = evaluate_loss_acc(model, loader, device); return l
    rows_metrics = [
        {"split": "train", "loss": loss_on(loaders["train"]), "acc": accuracy_score(train_y_true, train_y_pred), "f1_macro": f1_score(train_y_true, train_y_pred, average="macro")},
        {"split": "val",   "loss": loss_on(loaders["val"]),   "acc": accuracy_score(val_y_true,   val_y_pred),   "f1_macro": f1_score(val_y_true,   val_y_pred,   average="macro")},
        {"split": "test",  "loss": loss_on(loaders["test"]),  "acc": accuracy_score(test_y_true,  test_y_pred),  "f1_macro": f1_score(test_y_true,  test_y_pred,  average="macro")},
    ]
    write_metrics_csv(os.path.join(seed_root, "metrics.csv"), rows_metrics)

    # Reporte de clasificación y confusión en CSV (test)
    rep_dict = classification_report(test_y_true, test_y_pred, output_dict=True, zero_division=0)
    write_class_report_csv(os.path.join(seed_root, "classification_report.csv"), rep_dict)
    cm = confusion_matrix(test_y_true, test_y_pred); write_confusion_csv(os.path.join(seed_root, "confusion_matrix.csv"), cm)

    # Resumen seed
    n_params = int(sum(p.numel() for p in model.parameters()))
    with open(os.path.join(seed_root, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["key","value"])
        w.writerow(["best_epoch", int(stopper.best_epoch) if stopper.best_epoch >= 0 else len(curves)])
        w.writerow(["val_metric_name", cfg["train"]["val_metric"]])
        w.writerow(["val_metric_value", float(best_val_metric) if best_val_metric is not None else ""])
        w.writerow(["n_params", n_params])

    return {"seed": seed, "test_acc": rows_metrics[-1]["acc"], "test_f1": rows_metrics[-1]["f1_macro"], "best_epoch": int(stopper.best_epoch) if stopper.best_epoch >= 0 else len(curves)}

def main():
    cfg = CONFIG
    problem = cfg["run"]["problem"]; method = cfg["run"]["method_dir"]; outdir = cfg["run"]["outdir"]; run_id = now_run_id()
    run_root = os.path.join(outdir, problem, method, run_id); ensure_dir(run_root)

    print(f"[INFO] Run '{run_id}' | problema='{problem}' | método='{method}'")
    all_rows = []
    for seed in cfg["run"]["seeds"]:
        print(f"[INFO] Seed {seed}…")
        res = train_one_seed(seed, cfg, run_root); all_rows.append(res)

    # Aggregate (CSV)
    accs = [r["test_acc"] for r in all_rows]; f1s = [r["test_f1"] for r in all_rows]; best_epochs = [r["best_epoch"] for r in all_rows]
    with open(os.path.join(run_root, "aggregate.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem","method_dir","run_id","n_seeds","test_acc_mean","test_acc_std","test_f1_macro_mean","test_f1_macro_std","best_epoch_mean","best_epoch_std"])
        w.writerow([problem, method, run_id, len(all_rows),
                    float(np.mean(accs)), float(np.std(accs)),
                    float(np.mean(f1s)),  float(np.std(f1s)),
                    float(np.mean(best_epochs)), float(np.std(best_epochs))])

if __name__ == "__main__":
    main()
