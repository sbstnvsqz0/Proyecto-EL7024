# Proyecto-EL7024: Comparación de Standard vs. Information Dropout (Hito 3)

Este repositorio implementa un pipeline de experimentación para comparar **Standard Dropout** contra **Information Dropout** en un modelo MLP para el dataset MNIST.

En este **Hito 3**, el foco está en evaluar:
1.  **Robustez ante ruido**: Etiquetas ruidosas (Label Noise) y oclusión en las imágenes.
2.  **Compresión**: Análisis de la representación latente.

El proyecto está diseñado para ser un **pipeline automatizado**:
1.  **Configuración** centralizada mediante archivos `.yaml`.
2.  **Orquestación** de múltiples corridas (configuraciones x semillas) con un script principal.
3.  **Agregación** automática de resultados en un CSV final.

---

## 1. Ejecución

### Requisitos

* Python ≥ 3.9
* Librerías: disponibles en `requirements.txt`

```bash
git clone https://github.com/sbstnvsqz0/Proyecto-EL7024.git
cd Proyecto-EL7024
git checkout hito3
pip install -r requirements.txt
```

### Archivos Principales

*   `run_experiments.py` — **(Orquestador)** Script principal que lee una carpeta de `.yaml`, itera sobre todas las configuraciones y semillas, y ejecuta `src/hito3/train_and_test.py` para cada una. Finalmente, llama a `summarize_experiments.py`.
*   `src/hito3/train_and_test.py` — **(Ejecutor)** Ejecuta *un* solo experimento para *una* sola semilla. Carga la data, aplica transformaciones (incluyendo ruido), inicializa el `EngineMLP`, entrena, evalúa y guarda los artefactos.
*   `summarize_experiments.py` — **(Agregador)** Recolecta los resultados de *todas* las corridas, calcula estadísticas (media, std) y guarda un `summary_experiments.csv` global.
*   `src/hito3/engine.py` — **(Motor)** Define la clase `EngineMLP`, que encapsula la lógica de entrenamiento, validación, evaluación y el cálculo de la pérdida combinada (BCE + β*KL).
*   `src/hito3/paper_blocks.py` — **(Modelo)** Define la arquitectura del modelo, incluyendo `InformationDropout`, `MLPBlock` y `FullyConnectedPaper`.
*   `src/hito3/noises_transforms.py` — **(Ruido)** Implementa transformaciones de ruido para el dataset (ej. `DatasetLabelNoise`, `OcclusionNoise`).

### Estructura de Carpetas

La estructura del código fuente es:

```
.
│   run_experiments.py        # 1. El orquestador
│   summarize_experiments.py  # 3. El agregador
│   requirements.txt
│   README.md
│
├───experiments/              # ⚙️ Carpeta con las configuraciones (.yaml)
│   │   ...
│   └───hito3_32_compresion_noisy_oclussion_05/
│           info_beta_01.yaml
│           standard_p_05.yaml
│           ...
│
├───src/                      # 🧠 Código fuente
│   │   utils.py              # Utilidades generales
│   │
│   └───hito3/                # Código específico del Hito 3
│           engine.py
│           noises_transforms.py
│           paper_blocks.py
│           train_and_test.py # 2. El ejecutor
│
└───notebooks/                # 📊 Análisis y visualización
        data_visualization.ipynb
        hito3_results.ipynb
        pca_umap.ipynb
```

### Resultados

Al ejecutarse, se crea la siguiente estructura de **resultados** dentro de la carpeta del experimento:

```
experiments/{experiment_name}/
    results/
      {config_name}/
        models/
          {seed}.pth
        losses/
          losses_{seed}.csv
        plots/
          losses_plot_{seed}.png
        predictions/
          test_{seed}.csv
        summary_test.csv          # Resumen de métricas para esta config
    summary_experiments.csv       # <-- RESUMEN FINAL CONSOLIDADO
```

### Cómo Correr el Pipeline Completo

1.  **Configurar el experimento**:
    Edita `run_experiments.py` y modifica la lista `experiments_name` para que apunte a las carpetas de configuración deseada (ej. `hito3_32_compresion_noisy_oclussion_05`).

    ```python
    # En run_experiments.py
    experiments_name = ["hito3_32_compresion_noisy_oclussion_05"]
    ```
    Nota: En el repositorio se encuentran todas las configuraciones utilizadas en el hito final con algunas adicionales para las pruebas con ruidos.
2. **Flags Alternativas**:
    
    ```python
    # En run_experiments.py
    command = f"python -m src.hito3.train_and_test --experiment={yaml_path} --seed={seed}"
    ```
    Existen Flags adicionales para este comando como 
    - --get_histograms=True: Guarda histogramas de activaciones para todas las épocas del entrenamiento.
    - --train=False: Solo testea (se necesita haber entrenado antes).
    
    Estas flags se añaden al string command.
    
3.  **Ejecutar el orquestador**:

    ```bash
    python run_experiments.py
    ```

    Esto ejecutará secuencialmente todos los experimentos definidos en los `.yaml` de esa carpeta para las semillas especificadas.

---

## 2. Configuración (.yaml)

Los archivos `.yaml` en `experiments/` definen los hiperparámetros. Ejemplo:

```yaml
seeds: [0,1,2]

preprocessing_config:
  size: 24 #Tamaño original de set MNIST
  noise: 
    type: label  #gaussian | random_occlusion | contrast | label
    param: 0.1  #std | percentage | factor | percentage

model_config:
  hidden_dim: 32
  out_dim: 10
  dropout:
    type: information #"standard" | "information_static" |"information"
    p: 0.2
    initial_logvar: -5

train_config:
  epochs: 40
  batch_size: 128
  lr: 0.07
  optimizer: "sgd" #"adam" | "sgd"
  criterion: "cross_entropy"
  beta: 0.1
```
## 3. Resultados Hito Final

Las figuras del hito final se encuentran en el archivo `notebooks/hito3_results.ipynb`. Para ejecutarlo, se debe tener instalado jupyter notebook y las librerías especificadas en `requirements.txt`. Además se necesitan los resultados obtenidos del entrenamiento. A continuación se adjunta un link con las carpetas y resultados utilizados para el hito final: https://drive.google.com/drive/folders/1O1Wjyh6axxkI_i91Rk28DJripYpz4KWt?usp=sharing.


