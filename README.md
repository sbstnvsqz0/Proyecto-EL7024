````markdown
# Proyecto-EL7024: Comparación de Standard vs. Information Dropout (Hito 2)

Este repositorio implementa un pipeline de experimentación para comparar **Standard Dropout** contra **Information Dropout** (con varianza estática) en un modelo MLP para el dataset MNIST.

El proyecto está diseñado para ser un **pipeline automatizado**:
1.  **Configuración** centralizada mediante archivos `.yaml`.
2.  **Orquestación** de múltiples corridas (configuraciones x semillas) con un script principal.
3.  **Agregación** automática de resultados en un CSV final.

---

## 1) Ejecución

### Requisitos

* Python ≥ 3.9
* Librerías: disponibles en `requirements.txt`

```bash
pip install -r requirements.txt
````

### Archivos Principales

  * `run_experiments.py` — **(Orquestador)** Script principal que lee una carpeta de `.yaml`, itera sobre todas las configuraciones y semillas, y ejecuta `train_and_test.py` para cada una. Finalmente, llama a `summarize_experiments.py`.
  * `train_and_test.py` — **(Ejecutor)** Ejecuta *un* solo experimento para *una* sola semilla. Carga la data, inicializa el `EngineMLP`, entrena, evalúa y guarda los artefactos (modelo `.pth`, `losses.csv`, plots, etc.).
  * `summarize_experiments.py` — **(Agregador)** Recolecta los resultados de *todas* las corridas, calcula estadísticas (media, std) y guarda un `summary_experiments.csv` global.
  * `engine.py` — **(Motor)** Define la clase `EngineMLP`, que encapsula la lógica de entrenamiento, validación, evaluación, optimizador, scheduler y el cálculo de la pérdida combinada (BCE + β\*KL).
  * `paper_blocks.py` — **(Modelo)** Define la arquitectura del modelo, incluyendo `InformationDropout`, `MLPBlock` y `FullyConnectedPaper`.

### Estructura de Carpetas

La estructura del código fuente es:

```
.
│   run_experiments.py        # 1. El orquestador que se ejecuta
│   summarize_experiments.py  # 3. El agregador (llamado por el orquestador)
│   requirements.txt
│   README.md
│
├───experiments/              # ⚙️ Carpeta con las configuraciones
│   │   .gitignore
│   │
│   └───hito2_64_compresion_static/ # Un "banco" de experimentos
│           info_beta_0001.yaml     # Configuración para 1 experimento
│           info_beta_001.yaml
│           ...
│
├───src/                      # 🧠 Código fuente
│   │   utils.py              # Funciones (crear carpetas, plots)
│   │
│   └───hito2/
│           engine.py
│           paper_blocks.py
│           train_and_test.py # 2. El ejecutor (llamado por el orquestador)
│
└───notebooks/                # 📊 Análisis y visualización
        data_visualization.ipynb
```

Al ejecutarse, se crea la siguiente estructura de **resultados**:

```
experiments/
  {experiment_name}/              (ej. hito2_64_compresion_static)
    results/
      {config_name}/              (ej. info_beta_0001)
        models/
          {seed}.pth
        losses/
          losses_{seed}.csv
        plots/
          losses_plot_{seed}.png
          confusion_matrix_{seed}.png
        predictions/
          test_{seed}.csv
        summary_test.csv          # Resumen de métricas para esta config (agregado por seed)
    summary_experiments.csv       # <-- RESUMEN FINAL (agregado por config)
```

### Cómo Correr el Pipeline Completo

1.  **Configurar el "banco" de experimentos**:

      * Abre `run_experiments.py` y edita la variable `experiment_name` para que apunte a la carpeta de configuración que deseas ejecutar (ej. `hito2_64_compresion_static`).

    <!-- end list -->

    ```python
    # En run_experiments.py
    experiment_name = "hito2_64_compresion_static"  # <-- Edita esta línea
    ```

2.  **Ejecutar el orquestador**:

    ```bash
    python run_experiments.py
    ```

El script hará lo siguiente automáticamente:

1.  Iterará sobre cada `.yaml` en la carpeta `experiments/hito2_64_compresion_static/`.
2.  Para cada `.yaml`, leerá la lista de `seeds` y llamará a `python -m src.hito2.train_and_test ...` para cada semilla.
3.  Una vez que **todas** las corridas terminen, ejecutará `python summarize_experiments.py`.
4.  El resultado final consolidado aparecerá en `experiments/hito2_64_compresion_static/summary_experiments.csv`.

-----

## 2\) Propósito y Funcionamiento

### `run_experiments.py` (Orquestador)

  * Su única función es automatizar la ejecución en bucle.
  * Usa `subprocess.run()` para llamar a `train_and_test.py` como un módulo (`-m`). Esto asegura que cada corrida sea un proceso independiente.
  * Espera a que todos los subprocesos de entrenamiento terminen antes de lanzar el script de resumen.

### `train_and_test.py` (Ejecutor de 1 Run)

  * Es el "caballo de batalla" del pipeline. Está diseñado para ser llamado por el orquestador (o manualmente) con dos argumentos: `--experiment` (ruta al `.yaml`) y `--seed`.
  * **Responsabilidades**:
    1.  Leer el `.yaml` de configuración (`yaml.safe_load`).
    2.  Llamar a `create_folders` (de `utils.py`) para crear la estructura de salida (`results/.../{seed}`).
    3.  Cargar y pre-procesar el dataset (MNIST).
    4.  Instanciar la clase `EngineMLP` con las configuraciones de modelo y entrenamiento.
    5.  Si `train=True`, llama a `engine.train()`.
    6.  Llama a `engine.load_model()` para cargar el mejor checkpoint guardado.
    7.  Llama a `engine.evaluate()` en el conjunto de test.
    8.  Guarda los resultados de la evaluación (`results_per_seed`).

### `summarize_experiments.py` (Agregador)

  * Lee el argumento `--experiment` para saber qué carpeta de resultados analizar.
  * Busca todos los archivos `summary_test.csv` (que contienen los resultados por semilla de *una* configuración).
  * Concatena todos estos resúmenes.
  * Calcula métricas agregadas (media y std) para `accuracy`, `kl_loss`, `bce_loss`, etc.
  * Encuentra la `best_seed` (mejor semilla) basado en un criterio (ej. `accuracy` max o `kl_loss` min).
  * Guarda el `DataFrame` final como `summary_experiments.csv`.

### `engine.py` (Motor de ML)

  * Clase `EngineMLP` que contiene la lógica central de PyTorch.
  * **`__init__`**: Inicializa el modelo (`FullyConnectedPaper`), el optimizador (`Adam` o `SGD`), el `scheduler` (`MultiStepLR`) y el criterio (`CrossEntropyLoss`). Fija la semilla (`set_seed`).
  * **`train()`**: Contiene el bucle principal de entrenamiento y validación.
      * Calcula la pérdida combinada: `loss = self.criterion(output, y) + self.beta * kl_loss`.
      * Maneja `tqdm` para las barras de progreso.
      * Implementa **Early Stopping** implícito al guardar solo el mejor modelo (`self.save_dict()`) cuando `val_loss` disminuye.
  * **`evaluate()`**: Bucle de evaluación (sin `torch.no_grad()`) que calcula las pérdidas `bce_loss` y `kl_loss` por separado en el conjunto de test.
  * **`save_dict()` / `load_model()`**: Manejan el guardado y carga de checkpoints (`.pth`).
  * **`save_losses()`**: Guarda el CSV de `train_losses` y `val_losses` por época.

### `paper_blocks.py` (Arquitectura del Modelo)

  * **`InformationDropout`**: Implementación de Dropout Variacional con **varianza estática**. La `logvar` es un `nn.Parameter` (un vector de tamaño `input_features`) que se aprende, pero no depende de la entrada `x`.
  * **`InformationDropoutMLP`**: (No usada en tus configs, pero presente) Implementación con **varianza dinámica**, donde `logvar` es predicha por una `nn.Linear` que depende de `x`.
  * **`MLPBlock`**: Un bloque modular que aplica `Linear -> ReLU -> Dropout`. El bloque maneja la lógica de qué tipo de dropout usar ("standard" o "information\_static") y retorna `kl_loss = 0` si es standard.
  * **`FullyConnectedPaper`**: El modelo final. Es una secuencia de `MLPBlock` que suma las `kl_loss` de cada capa (aunque en tu implementación actual solo usas una capa con KL).

<!-- end list -->
