# Proyecto-EL7024

**Information-Theoretic Dropout in Neural Networks**

## 1) Ejecución

### Requisitos

* Python ≥ 3.9
* Librerías: `torch`, `numpy`, `scikit-learn`, `pandas`, `matplotlib`, `PyYAML`

```bash
pip install torch numpy scikit-learn pandas matplotlib PyYAML
```

### Archivos principales

* `problem.py` — define y registra problemas de estimación (incluye **moons** y **blobs**).
* `solver.py` — entrena varios MLP con parámetros definidos en archivos .yaml, **solo CSVs** (sin gráficos).
* `benchmark.py` — **genera plots y resúmenes** a partir de los CSVs (curvas, confusión, scatter 2D).

### Estructura de carpetas (se crea automáticamente)

```
results/
  {problem}/
    {method_dir}/
      {run_id}/
        seed_{k}/
          args.csv
          curves.csv
          metrics.csv
          summary.csv
          classification_report.csv
          confusion_matrix.csv
          predictions_train.csv
          predictions_val.csv
          predictions_test.csv
        aggregate.csv

benchmarks_plots/
  {method_dir}/
    {problem}/
      {run_id}/
        curves_aggregate.csv
        curves_aggregate.png
        confusion_best_seed.png
        scatter_moons_errors.png   # si problem=moons (o blobs en 2D)
        test_acc_per_seed.png
        aggregate_copy.csv
```

### Configuración (sin flags)

Todos los parámetros se editan **dentro del .yaml**:

* En `solve_simple.py` modifica el dict `CONFIG`:

  * `CONFIG["run"]`: `problem` (`"moons"` o `"blobs"`), `method_dir`, `seeds`, etc.
  * `CONFIG["data"]`: tamaño de dataset, ruido, splits, `batch_size`, etc.
  * `CONFIG["model"]`: `hidden_sizes`, `activation`.
  * `CONFIG["train"]`: `epochs`, `optimizer`, `lr`, `patience`, `val_metric`.
* En `benchmark.py` modifica `BENCH_CFG`:

  * `root`, `problems`, `method_dirs`, carpeta de salida `out`.
  * `best_seed_metric` para elegir el mejor seed por `val_loss` o `val_acc`.

### Cómo correr

1. **Entrenamiento baseline (solo CSVs):**

```bash
python solve_simple.py
```

Genera, por `seed`, CSVs con:

* `args.csv` (parámetros del run)
* `curves.csv` (por época: `train_loss`, `val_loss`, `train_acc`, `val_acc`)
* `metrics.csv` (por split: `loss`, `acc`, `f1_macro`)
* `classification_report.csv`, `confusion_matrix.csv`
* `predictions_*.csv` (pares `y_true`, `y_pred` por split)
* `summary.csv` (mejor época, métrica de validación, #params)

Y un `aggregate.csv` a nivel de run (promedios ± std entre seeds).

2. **Benchmark + gráficos:**

```bash
python benchmark.py
```

Produce:

* **Curvas agregadas** con banda de desviación: `curves_aggregate.(csv|png)`
* **Matriz de confusión** del mejor seed: `confusion_best_seed.png`
* **Scatter 2D** de **moons** (o blobs 2D) con **errores del test** remarcados: `scatter_moons_errors.png`
* **Barras de accuracy por seed**: `test_acc_per_seed.png`
* Copia del `aggregate.csv`: `aggregate_copy.csv`

> Para añadir otro método (p. ej., `solve_dropout.py`), usa el **mismo contrato de salida** (CSV con los mismos nombres/columnas) y un `method_dir` distinto (ej. `dropout_p0.2`). Luego agrega ese `method_dir` en `BENCH_CFG["method_dirs"]`.

---

## 2) Propósito y funcionamiento (funciones clave)

### `problem.py`

* **`ProblemSpec`** (dataclass): especificación de un problema.

  * `name`, `task`, `n_features`, `n_classes`, `build_datasets`, `default_*`, `plotting`.
* **Registro `PROBLEMS`**: diccionario con problemas disponibles.
* **`get_problem(name)`**: devuelve la `ProblemSpec` registrada.
* **`build_moons(cfg)` / `build_blobs(cfg)`**:

  * Generan datos sintéticos (scikit-learn), dividen en **train/val/test** con `train_test_split`.
  * **Estandarizan** usando estadísticas de *train* (vía `StandardScaler`).
  * Devuelven: `{"loaders": …, "n_features", "n_classes", "X_vis", "y_vis", "class_names", "plotting"}`.

    * `X_vis`/`y_vis` (ya estandarizados) permiten graficar fronteras o scatter 2D en el **benchmark**.
* **Privadas**: `_standardize_split`, `_to_loaders`.

  * Aseguran que los `DataLoader`s de validación y test **no se barajen**, para alinear con las predicciones guardadas.

### `solve_simple.py` (baseline **sin dropout**, **solo CSVs**)

* **CONFIG**: único punto de edición (sin CLI). Define problema, datos, modelo y entrenamiento.
* **`SimpleMLP`**: MLP secuencial lineal + activación; **no** incluye Dropout.
* **Entrenamiento**:

  * **`train_one_seed`**: fija semillas, construye datasets, instancia el modelo y entrena con **early stopping** (`val_loss` o `val_acc`).
  * Guarda por época en `curves.csv` y, al final, métricas por split en `metrics.csv`.
  * Exporta `predictions_*.csv` (necesarias para los gráficos del benchmark).
  * `classification_report.csv` y `confusion_matrix.csv` se derivan del split test.
  * `summary.csv` almacena: mejor época, métrica de validación y número de parámetros.
* **Helpers**:

  * `evaluate_loss_acc`, `predict_all` (evaluación/predicción batcheada).
  * `EarlyStopping` (lógica de parada temprana).
  * `write_*_csv` (salidas tabulares homogéneas).

### `benchmark.py` (plots + agregación)

* **BENCH_CFG**: raíz de resultados (`root`), `problems`, `method_dirs`, carpeta de salida y métrica para elegir el mejor seed.
* **Descubrimiento y carga**:

  * `find_runs` (encuentra `run_id`s), lectura de `curves.csv`, `metrics.csv`, `summary.csv`, `predictions_test.csv`.
* **Selección del mejor seed**:

  * `choose_best_seed`: lee `summary.csv` por seed y escoge el que tenga la mejor `val_metric_value` según `val_metric_name`.
* **Agregación de curvas**:

  * `aggregate_curves`: promedia (y calcula std) `train/val loss/acc` por época, luego genera `curves_aggregate.(csv|png)`.
* **Gráficos**:

  * `plot_curves_aggregate`: 2 paneles (loss y accuracy) con bandas de ±1σ.
  * `plot_confusion_from_csv`: reconstruye la matriz desde `confusion_matrix.csv` y la dibuja con anotaciones.
  * `scatter_moons_with_errors`:

    * Reconstruye el dataset estandarizado **con la misma semilla y parámetros** (usando `problem.get_problem` y `args.csv`).
    * Carga `predictions_test.csv`, marca **muestras mal clasificadas** (`x` negras) sobre el **scatter** de todos los puntos (coloreados por etiqueta verdadera).
  * `bar_seed_accuracies`: barras de `accuracy` en test para cada seed del run.
* **Salidas**:

  * CSV y PNG en `benchmarks_plots/{method}/{problem}/{run_id}/`, listos para informe.

---

**Notas**

* El baseline reproduce el comparador **“No dropout”** del paper; es la línea base para luego integrar un `solve_dropout.py` (Dropout clásico) o un cuello tipo IB/VIB.
* Para **moons**/**blobs** con más de 2 features, el scatter 2D usa las dos primeras dimensiones estandarizadas; si no es 2D, el benchmark omite el scatter.
* Si quieres comparar varios métodos, corre cada uno con un `method_dir` distinto y añádelo a `BENCH_CFG["method_dirs"]` para graficarlos/organizarlos por separado.
