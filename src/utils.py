import numpy as np
import torch, random, os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_criterion(criterion:str):
    from torch.nn import CrossEntropyLoss
    assert criterion in ["cross_entropy"]
    return CrossEntropyLoss()


def create_folders(exp_file:str):
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

def save_losses(losses_dict:dict,losses_dir:str, plots_dir:str, seed:int):
    assert "train_losses" in losses_dict.keys() and "val_losses" in losses_dict.keys(), "keys de diccionario losses incompleto"
    assert len(losses_dict["train_losses"])==len(losses_dict["val_losses"]), "losses tienen largos distintos"
    print(f"Saving losses csv in {losses_dir}")
    df = pd.DataFrame(losses_dict)
    df.to_csv(os.path.join(losses_dir,f"losses_{seed}.csv"),index=False)
    print("Done!")
    print(f"Saving losses plot in {plots_dir}")
    plot_losses(losses_dict=losses_dict,plots_dir=plots_dir,seed=seed,save=True)
    print("Done!")


    
def plot_confusion_matrix(real_values, pred_values, save_dir,seed,set_name,save=True,figsize=(8, 6)):

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
        plt.savefig(os.path.join(save_dir,f"confusion_matrix_{seed}.png"))
        plt.close()
    else:
        plt.show()
    

def results_per_seed(results_dict:dict,preds_save_dir:str,plots_save_dir:str,seed:int,set_name:str):
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


