# solve_simple.py
from __future__ import annotations
import os, csv
from typing import Dict, Any

import numpy as np
import yaml
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from problem import get_problem, ProblemSpec
from src.blocks import MLP
from utils.utils_solver import *
import argparse

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
    model = MLP(n_features, cfg["model"]["hidden_sizes"], n_classes, cfg["model"]["activation"],cfg["model"]["dropout"]["type"],cfg["model"]["dropout"]["dropout_prob"]).to(device)

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
        "problem": cfg["run"]["problem"], "method": run_root.split(os.sep)[-1] , "seed": seed,
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
    
    parser = argparse.ArgumentParser(prog="solver")
    parser.add_argument('--exp',type=str,default="example")

    args = parser.parse_args()
    experiment = args.exp
    folder_exp = os.path.join("experiments",experiment)
    
    yamls_exp = list(filter(lambda x: x.endswith(".yaml"),os.listdir(folder_exp)))
    yaml_paths = [os.path.join(folder_exp,yaml_exp) for yaml_exp in yamls_exp]
    for yaml_path in yaml_paths:
        with open(yaml_path, "r") as f:
            cfg: Dict[str, Any] = yaml.safe_load(f)
        name = yaml_path.split(os.sep)[-1].split(".")[0]
        problem = cfg["run"]["problem"]; method = name; outdir = cfg["run"]["outdir"]
        run_root = os.path.join(outdir, experiment,problem, method); ensure_dir(run_root)

        print(f"[INFO] Run '{name}' | problema='{problem}' | método='{method}'")
        all_rows = []
        for seed in cfg["run"]["seeds"]:
            print(f"[INFO] Seed {seed}…")
            res = train_one_seed(seed, cfg, run_root); all_rows.append(res)

        # Aggregate (CSV)
        accs = [r["test_acc"] for r in all_rows]; f1s = [r["test_f1"] for r in all_rows]; best_epochs = [r["best_epoch"] for r in all_rows]
        with open(os.path.join(run_root, "aggregate.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["problem","method_dir","n_seeds","test_acc_mean","test_acc_std","test_f1_macro_mean","test_f1_macro_std","best_epoch_mean","best_epoch_std"])
            w.writerow([problem, method, len(all_rows),
                        float(np.mean(accs)), float(np.std(accs)),
                        float(np.mean(f1s)),  float(np.std(f1s)),
                        float(np.mean(best_epochs)), float(np.std(best_epochs))])

if __name__ == "__main__":
    main()
