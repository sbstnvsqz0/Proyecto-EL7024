# benchmark.py
from __future__ import annotations
import os, glob, csv
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from problem import get_problem

# =========================
# CONFIGURACIÓN EN CÓDIGO
# =========================
BENCH_CFG: Dict[str, Any] = {
    "root": "results",
    "problems": ["moons", "blobs"],                     # ajusta según lo corrido
    "method_dirs": ["simple_no_dropout_csv"],           # añade otros métodos si quieres
    "out": "benchmarks_plots",
    # para escoger el "mejor seed" por run
    "best_seed_metric": "val_loss",   # "val_loss" (mínimo) o "val_acc" (máximo) — debe coincidir con solve
}

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def find_runs(root: str, problem: str, method_dir: str) -> List[str]:
    base = os.path.join(root, problem, method_dir)
    if not os.path.isdir(base): return []
    runs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]
    runs.sort()
    return runs

def read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path): return None
    return pd.read_csv(path)

def read_kv_csv(path: str) -> Dict[str, Any]:
    out = {}
    if not os.path.exists(path): return out
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        out[str(row["key"])] = row["value"]
    return out

def choose_best_seed(seed_dirs: List[str], prefer_metric: str = "val_loss") -> str | None:
    """
    Lee summary.csv en cada seed y escoge la que tenga mejor val_metric_value
    según el nombre en 'val_metric_name'. Si 'prefer_metric' difiere del guardado,
    se usa el guardado.
    """
    best_dir, best_val = None, None
    for sd in seed_dirs:
        s = read_csv(os.path.join(sd, "summary.csv"))
        if s is None or s.empty: continue
        kv = {row["key"]: row["value"] for _, row in s.iterrows()}
        name = kv.get("val_metric_name", prefer_metric)
        val  = kv.get("val_metric_value", None)
        if val is None or val == "": continue
        val = float(val)
        if name == "val_loss":  # queremos mínimo
            better = (best_val is None) or (val < best_val)
        else:  # val_acc -> máximo
            better = (best_val is None) or (val > best_val)
        if better:
            best_val = val; best_dir = sd
    return best_dir

def aggregate_curves(seed_dirs: List[str]) -> pd.DataFrame | None:
    dfs = []
    for sd in seed_dirs:
        c = read_csv(os.path.join(sd, "curves.csv"))
        if c is None or c.empty: continue
        c = c.copy(); c["seed_dir"] = os.path.basename(sd); dfs.append(c)
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    agg = df.groupby("epoch", as_index=False).agg({
        "train_loss": ["mean","std"],
        "val_loss":   ["mean","std"],
        "train_acc":  ["mean","std"],
        "val_acc":    ["mean","std"]
    })
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]
    # renombrar epoch_
    agg = agg.rename(columns={"epoch_": "epoch"})
    return agg

def plot_curves_aggregate(curves_agg: pd.DataFrame, out_png: str):
    fig = plt.figure(figsize=(9, 4.6))
    # Loss
    ax1 = plt.subplot(1, 2, 1)
    x = curves_agg["epoch"].values
    for key in [("train_loss_mean","train_loss_std","train_loss"),
                ("val_loss_mean","val_loss_std","val_loss")]:
        m = curves_agg[key[0]].values; s = curves_agg[key[1]].fillna(0).values
        ax1.plot(x, m, label=key[2])
        ax1.fill_between(x, m - s, m + s, alpha=0.2)
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    # Acc
    ax2 = plt.subplot(1, 2, 2)
    for key in [("train_acc_mean","train_acc_std","train_acc"),
                ("val_acc_mean","val_acc_std","val_acc")]:
        m = curves_agg[key[0]].values; s = curves_agg[key[1]].fillna(0).values
        ax2.plot(x, m, label=key[2])
        ax2.fill_between(x, m - s, m + s, alpha=0.2)
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_confusion_from_csv(csv_path: str, out_png: str):
    df = read_csv(csv_path)
    if df is None or df.empty: return
    # reconstruir matriz
    cols = [c for c in df.columns if c.startswith("pred_")]
    cm = df[cols].values.astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([c.replace("pred_","") for c in cols], rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in range(len(cols))])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def scatter_moons_with_errors(seed_dir: str, args_kv: Dict[str, Any], out_png: str):
    """
    Reconstruye los datos estandarizados (X_vis, y_vis) y el split de test
    con la misma semilla y parámetros, y sobrepone mal clasificados del test.
    """
    # reconstruir config
    problem = args_kv.get("problem", "moons")
    seed = int(args_kv.get("seed", 0))
    # parámetros de data
    data_cfg = {
        "seed": seed,
        "n_samples": int(float(args_kv.get("n_samples", 4000))) if args_kv.get("n_samples","") != "" else 4000,
        "noise": float(args_kv.get("noise", 0.2)) if args_kv.get("noise","") != "" else 0.2,
        "centers": int(float(args_kv.get("centers", 4))) if args_kv.get("centers","") != "" else 4,
        "cluster_std": float(args_kv.get("cluster_std", 1.5)) if args_kv.get("cluster_std","") != "" else 1.5,
        "val_size": float(args_kv.get("val_size", 0.2)),
        "test_size": float(args_kv.get("test_size", 0.2)),
        "batch_size": int(float(args_kv.get("batch_size", 128))),
        "stratify": True if str(args_kv.get("stratify","True")).lower() == "true" else False
    }
    spec = get_problem(problem)
    built = spec.build_datasets(data_cfg)

    # puntos totales en espacio estandarizado
    X_vis = built.get("X_vis", None)
    y_vis = built.get("y_vis", None)
    if X_vis is None or y_vis is None or X_vis.shape[1] < 2:
        return  # no graficar si no es 2D

    # Extraer test set en el MISMO orden del loader (shuffle=False en problem.py)
    test_loader = built["loaders"]["test"]
    X_test = []; y_test = []
    for xb, yb in test_loader:
        X_test.append(xb.numpy()); y_test.append(yb.numpy())
    X_test = np.vstack(X_test); y_test = np.concatenate(y_test)

    # Cargar predicciones de test (generadas por solve_simple.py)
    pred_path = os.path.join(seed_dir, "predictions_test.csv")
    pdf = read_csv(pred_path)
    if pdf is None or pdf.empty: return
    y_pred = pdf["y_pred"].values.astype(int)

    # Máscara de errores
    mis = (y_pred != y_test)

    # Plot
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    sc = ax.scatter(X_vis[:,0], X_vis[:,1], c=y_vis, s=10, alpha=0.5, edgecolor="none", cmap="tab10")
    # Sobreponer mal clasificados del test
    ax.scatter(X_test[mis,0], X_test[mis,1], marker="x", s=45, linewidths=1.5, c="k", label="Misclassified (test)")
    ax.set_title("Moons — standardized space\nAll points (true labels) + misclassified test samples")
    ax.set_xlabel("x1 (std)"); ax.set_ylabel("x2 (std)")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def bar_seed_accuracies(seed_dirs: List[str], out_png: str):
    vals = []
    for sd in seed_dirs:
        m = read_csv(os.path.join(sd, "metrics.csv"))
        if m is None or m.empty: continue
        row = m.loc[m["split"] == "test"]
        if row.empty: continue
        vals.append(float(row.iloc[0]["acc"]))
    if not vals: return
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.bar(range(len(vals)), vals)
    ax.set_title("Test accuracy per seed")
    ax.set_xlabel("seed idx (order of discovery)"); ax.set_ylabel("accuracy")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def main():
    cfg = BENCH_CFG
    root = cfg["root"]; out_root = cfg["out"]
    ensure_dir(out_root)

    for method_dir in cfg["method_dirs"]:
        method_out = os.path.join(out_root, method_dir); ensure_dir(method_out)

        for problem in cfg["problems"]:
            runs = find_runs(root, problem, method_dir)
            if not runs:
                print(f"[WARN] No runs en {root}/{problem}/{method_dir}")
                continue

            for run_dir in runs:
                run_id = os.path.basename(run_dir)
                out_run = os.path.join(method_out, problem, run_id); ensure_dir(out_run)

                # Descubrir seeds
                seed_dirs = sorted([d for d in glob.glob(os.path.join(run_dir, "seed_*")) if os.path.isdir(d)])
                if not seed_dirs: continue

                # 1) Curvas agregadas
                curves_agg = aggregate_curves(seed_dirs)
                if curves_agg is not None:
                    curves_agg.to_csv(os.path.join(out_run, "curves_aggregate.csv"), index=False)
                    plot_curves_aggregate(curves_agg, os.path.join(out_run, "curves_aggregate.png"))

                # 2) Matriz de confusión del mejor seed
                best_seed = choose_best_seed(seed_dirs, prefer_metric=cfg["best_seed_metric"])
                if best_seed is not None:
                    plot_confusion_from_csv(os.path.join(best_seed, "confusion_matrix.csv"),
                                            os.path.join(out_run, "confusion_best_seed.png"))

                # 3) Scatter 2D para moons/blobs 2D mostrando errores del test (usa args.csv + problem.py)
                # Tomamos cualquier seed (la primera) solo para recuperar args y reconstruir splits
                args_kv = read_kv_csv(os.path.join(seed_dirs[0], "args.csv"))
                if problem == "moons":
                    scatter_moons_with_errors(best_seed or seed_dirs[0], args_kv, os.path.join(out_run, "scatter_moons_errors.png"))
                elif problem == "blobs":
                    # (opcional) repetir la misma lógica para blobs 2D
                    scatter_moons_with_errors(best_seed or seed_dirs[0], args_kv, os.path.join(out_run, "scatter_blobs_errors.png"))

                # 4) Barras de accuracy por seed
                bar_seed_accuracies(seed_dirs, os.path.join(out_run, "test_acc_per_seed.png"))

                # 5) Resumen CSV del run (desde aggregate.csv si existe)
                agg_csv = os.path.join(run_dir, "aggregate.csv")
                if os.path.exists(agg_csv):
                    # Copia a carpeta del benchmark por conveniencia
                    df = pd.read_csv(agg_csv)
                    df.to_csv(os.path.join(out_run, "aggregate_copy.csv"), index=False)

if __name__ == "__main__":
    main()
