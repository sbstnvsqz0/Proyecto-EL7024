import numpy as np
import torch, random, os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def set_seed(seed:int):
    """
    Fija la semilla para la reproducibilidad en random, numpy y torch.
    
    Args:
        seed (int): Semilla a utilizar.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_auto() -> torch.device:
    """
    Retorna el dispositivo disponible (CPU o CUDA).
    
    Returns:
        torch.device: Dispositivo a utilizar.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_criterion(criterion:str):
    """
    Configura y retorna la función de pérdida especificada.
    
    Args:
        criterion (str): Nombre del criterio (ej. "cross_entropy").
        
    Returns:
        nn.Module: Instancia de la función de pérdida.
    """
    from torch.nn import CrossEntropyLoss
    assert criterion in ["cross_entropy"]
    return CrossEntropyLoss()


def create_folders(exp_file:str):
    """
    Crea la estructura de directorios para guardar los resultados del experimento.
    
    Args:
        exp_file (str): Ruta al archivo de configuración .yaml.
        
    Returns:
        tuple: Rutas a las carpetas creadas (experiments, models, losses, predictions, plots).
    """
    #Se crea carpeta results/nombre_de_experimento
    exp_name = exp_file.split(os.sep)[-1].split(".")[0]
    folder_experiments = os.path.join(os.sep.join(exp_file.split(os.sep)[:-1]),"results",exp_name)
    os.makedirs(folder_experiments,exist_ok=True)
    #Se crea carpeta results/nombre_de_experimento/models
    model_save_dir = os.path.join(folder_experiments,"models")
    os.makedirs(model_save_dir,exist_ok=True)
    #Se crea carpeta results/nombre_de_experimento/losses
    losses_save_dir = os.path.join(folder_experiments,"losses")
    os.makedirs(losses_save_dir,exist_ok=True)
    #Se crea carpeta results/nombre_de_experimento/predictions
    preds_save_dir = os.path.join(folder_experiments,"predictions")
    os.makedirs(preds_save_dir,exist_ok=True)
    #Se crea carpeta results/nombre_de_experimento/plots
    plots_save_dir = os.path.join(folder_experiments,"plots")
    os.makedirs(plots_save_dir,exist_ok=True)
    return folder_experiments,model_save_dir,losses_save_dir,preds_save_dir,plots_save_dir

def plot_losses(losses_dict:dict,plots_dir:str,seed:int,save:bool=True):
    """
    Genera y guarda (o muestra) el gráfico de las pérdidas de entrenamiento y validación.
    
    Args:
        losses_dict (dict): Diccionario con las listas de pérdidas.
        plots_dir (str): Directorio donde guardar el gráfico.
        seed (int): Semilla del experimento.
        save (bool): Si es True guarda el archivo, si es False lo muestra.
    """
    assert "train_losses" in losses_dict.keys() and "val_losses" in losses_dict.keys(), "keys de diccionario losses incompleto"
    assert len(losses_dict["train_losses"])==len(losses_dict["val_losses"]), "losses tienen largos distintos"
    lenght = len(losses_dict["train_losses"])
    plt.figure(figsize=(10, 5))
    plt.plot(range(lenght), losses_dict["train_losses"], label="Train Loss")
    plt.plot(range(lenght), losses_dict["val_losses"], label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    title = f"Losses\nseed {seed}"
    plt.title(title)
    if save:
        plt.savefig(os.path.join(plots_dir,f"losses_plot_{seed}.png"))
        plt.close()
    else:
        plt.show()
        
def plot_kl_losses(losses_dict:dict,plots_dir:str,seed:int,save:bool=True):
    """
    Genera y guarda (o muestra) el gráfico de las pérdidas KL de entrenamiento y validación.
    
    Args:
        losses_dict (dict): Diccionario con las listas de pérdidas KL.
        plots_dir (str): Directorio donde guardar el gráfico.
        seed (int): Semilla del experimento.
        save (bool): Si es True guarda el archivo, si es False lo muestra.
    """
    assert "train_kl_losses" in losses_dict.keys() and "val_kl_losses" in losses_dict.keys(), "keys de diccionario losses incompleto"
    assert len(losses_dict["train_kl_losses"])==len(losses_dict["val_kl_losses"]), "losses tienen largos distintos"
    lenght = len(losses_dict["train_kl_losses"])
    plt.figure(figsize=(10, 5))
    plt.plot(range(lenght), losses_dict["train_kl_losses"], label="Train KL Loss")
    plt.plot(range(lenght), losses_dict["val_kl_losses"], label="Val KL Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    title = f"KL Losses\nseed {seed}"
    plt.title(title)
    if save:
        plt.savefig(os.path.join(plots_dir,f"kl_losses_plot_{seed}.png"))
        plt.close()
    else:
        plt.show()

def save_losses(losses_dict:dict,losses_dir:str, plots_dir:str, seed:int):
    """
    Guarda las pérdidas en un archivo CSV y genera los gráficos correspondientes.
    
    Args:
        losses_dict (dict): Diccionario con las pérdidas.
        losses_dir (str): Directorio donde guardar el CSV.
        plots_dir (str): Directorio donde guardar los gráficos.
        seed (int): Semilla del experimento.
    """
    assert "train_losses" in losses_dict.keys() and "val_losses" in losses_dict.keys(), "keys de diccionario losses incompleto"
    assert len(losses_dict["train_losses"])==len(losses_dict["val_losses"]), "losses tienen largos distintos"
    assert "train_kl_losses" in losses_dict.keys() and "val_kl_losses" in losses_dict.keys(), "keys de diccionario losses incompleto"
    assert len(losses_dict["train_kl_losses"])==len(losses_dict["val_kl_losses"]), "kl losses tienen largos distintos"
    print(f"Saving losses csv in {losses_dir}")
    df = pd.DataFrame(losses_dict)
    df.to_csv(os.path.join(losses_dir,f"losses_{seed}.csv"),index=False)
    print("Done!")
    print(f"Saving losses plot in {plots_dir}")
    plot_losses(losses_dict=losses_dict,plots_dir=plots_dir,seed=seed,save=True)
    print("Done!")
    print(f"Saving kl losses plot in {plots_dir}")
    plot_kl_losses(losses_dict=losses_dict,plots_dir=plots_dir,seed=seed,save=True)
    print("Done!")

    
def plot_confusion_matrix(real_values, pred_values, save_dir,seed,set_name,save=True,figsize=(8, 6)):
    """
    Genera y guarda (o muestra) la matriz de confusión.
    
    Args:
        real_values (array-like): Etiquetas reales.
        pred_values (array-like): Etiquetas predichas.
        save_dir (str): Directorio donde guardar el gráfico.
        seed (int): Semilla del experimento.
        set_name (str): Nombre del conjunto (ej. "test").
        save (bool): Si es True guarda el archivo, si es False lo muestra.
        figsize (tuple): Tamaño de la figura.
    """
    cm = confusion_matrix(real_values, pred_values)
    
    all_labels = np.unique(np.concatenate((real_values, pred_values)))
    class_names = [str(label) for label in sorted(all_labels)]
     
    plt.figure(figsize=figsize)
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar=False) 
    title = f"Confusion Matrix in {set_name}\n seed {seed}"
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_dir,f"confusion_matrix_{set_name}_{seed}.png"))
        plt.close()
    else:
        plt.show()
    

def results_per_seed(results_dict:dict,preds_save_dir:str,plots_save_dir:str,summary_dir:str,seed:int,set_name:str,n_seeds:int):
    """
    Guarda las predicciones, matriz de confusión y métricas resumen para una semilla específica.
    
    Args:
        results_dict (dict): Diccionario con resultados (etiquetas reales, predichas, accuracy, losses).
        preds_save_dir (str): Directorio para guardar predicciones.
        plots_save_dir (str): Directorio para guardar gráficos.
        summary_dir (str): Directorio para guardar el resumen.
        seed (int): Semilla del experimento.
        set_name (str): Nombre del conjunto (ej. "test").
        n_seeds (int): Número total de semillas (para lógica de append en CSV).
    """
    print(f"Saving preds in {preds_save_dir}")
    df_preds = pd.DataFrame({"real_label":results_dict["real_labels"],
                            "pred_label":results_dict["pred_labels"]})
    df_preds.to_csv(os.path.join(preds_save_dir,f"{set_name}_{seed}.csv"),index=False)
    print("Done!")
    print(f"Saving confusion matrix in {plots_save_dir}")
    plot_confusion_matrix(real_values=results_dict["real_labels"],
                        pred_values = results_dict["pred_labels"], 
                        save_dir = plots_save_dir,
                        seed = seed,
                        set_name=set_name,
                        save=True,
                        figsize=(8, 6))
    print("Done!")
    print(f"Saving summary results in {summary_dir}")
    df_summary = pd.DataFrame({ "seed":[seed],
                                "accuracy":[results_dict["accuracy"]],
                               "kl_loss":[results_dict["kl_loss"]],
                               "bce_loss":[results_dict["bce_loss"]],
                               "total_loss":[results_dict["kl_loss"]+results_dict["bce_loss"]]})
    summary_path = os.path.join(summary_dir,f"summary_{set_name}.csv")
    if not os.path.isfile(summary_path):
        df_summary.to_csv(summary_path,index=False)
    else:
        if len(pd.read_csv(summary_path)) >= n_seeds:
            df_summary.to_csv(summary_path,index=False)
        else:
            df_existing = pd.read_csv(summary_path)
            df_updated = pd.concat([df_existing,df_summary],ignore_index=True)
            df_updated.to_csv(summary_path,index=False)
    print("Done!")


def create_stratified_subset(seed, dataset, subset_size=0.1):
        """
        Crea un subset del dataset manteniendo la proporción de etiquetas.
        subset_size: Flotante (0.0 a 1.0) representando la fracción del dataset a usar.
        """

        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        else:
            labels = [y.item() for _, y in DataLoader(dataset, batch_size=1, num_workers=0)]
            

        indices = np.arange(len(dataset))
        
        subset_indices, _ = train_test_split(
            indices, 
            train_size=subset_size, 
            stratify=labels, 
            random_state=seed,
        )
        
        return Subset(dataset, subset_indices)


