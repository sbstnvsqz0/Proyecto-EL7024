from .paper_blocks import FullyConnectedPaper
from typing import Dict, Any
import torch
from tqdm import tqdm
from src.utils import (set_seed, device_auto,set_criterion, save_losses)
import os
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from torch.optim import Adam, SGD
from matplotlib import pyplot as plt

class EngineMLP:
    """
    Clase encargada del entrenamiento, validación y evaluación del modelo MLP.
    """
    def __init__(self, seed, save_model_dir, save_losses_dir, save_plots_dir, mlp_config: Dict[str, Any], train_config: Dict[str, Any]):
        """
        Inicializa el motor de entrenamiento.

        Args:
            seed (int): Semilla para reproducibilidad.
            save_model_dir (str): Directorio para guardar el modelo.
            save_losses_dir (str): Directorio para guardar las pérdidas.
            save_plots_dir (str): Directorio para guardar los gráficos.
            mlp_config (Dict[str, Any]): Configuración del modelo MLP.
            train_config (Dict[str, Any]): Configuración del entrenamiento (epochs, lr, etc.).
        """
        self.seed = seed
        set_seed(self.seed)
        self.device = device_auto()
        self.save_model_dir = save_model_dir
        self.save_losses_dir = save_losses_dir
        self.save_plots_dir = save_plots_dir
        self.model = FullyConnectedPaper(**mlp_config).to(self.device)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.lr = train_config["lr"]
        self.criterion = set_criterion(train_config["criterion"])
        self.optimizer = Adam(self.model.parameters(),lr=self.lr) if train_config["optimizer"]=="adam" else SGD(self.model.parameters(),lr=self.lr,momentum=0.9) #Paper usa SGD con momentum
        self.scheduler = MultiStepLR(self.optimizer, milestones=[15,35], gamma=0.1) #Entrenamiento en paper disminuye learning rate en 0.1 en épocas 30 y 70; se entrena la mitad de epocas, entonces se dividieron por 2
        self.beta = train_config["beta"]
        

        self.best_val_loss = float('inf')
        self.train_loss = []
        self.val_loss = []
        self.train_kl_loss = []
        self.val_kl_loss = []

    def train(self,train_dataset,val_dataset,histogram_dataset):
        """
        Ejecuta el ciclo de entrenamiento y validación.

        Args:
            train_dataset (Dataset): Dataset de entrenamiento.
            val_dataset (Dataset): Dataset de validación.
            histogram_dataset (Dataset): Dataset para generar histogramas de activaciones (opcional).
        """
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=4)
        epoch_pbar = tqdm(range(self.epochs), desc="Epochs")

        for epoch in epoch_pbar:
            self.model.train()
            train_loss = 0.0
            train_acc = 0
            train_epoch_kl_loss = 0.0
            total = 0
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
            for x, y in train_pbar:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output, kl_loss = self.model(x)
                loss = self.criterion(output, y) + self.beta*kl_loss
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * x.size(0)
                train_epoch_kl_loss += kl_loss * x.size(0)
                train_acc += (output.argmax(dim=1) == y).sum().item()
                total += x.size(0) 
            self.scheduler.step()
            train_loss /= total
            train_acc /= total
            train_epoch_kl_loss /= total
            self.train_loss.append(train_loss)
            try:
                self.train_kl_loss.append(train_epoch_kl_loss.item())
            except: 
                self.train_kl_loss.append(train_epoch_kl_loss)

            self.model.eval()
            val_loss = 0.0
            val_acc = 0
            val_epoch_kl_loss = 0.0
            total = 0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc="Validation", leave=False):
                    x, y = x.to(self.device), y.to(self.device)
                    output, kl_loss = self.model(x)
                    loss = self.criterion(output, y) + self.beta*kl_loss
                    val_loss += loss.item() * x.size(0)
                    val_epoch_kl_loss += kl_loss * x.size(0)
                    val_acc += (output.argmax(dim=1) == y).sum().item()
                    total += x.size(0)
            val_loss /= total
            val_acc /= total
            val_epoch_kl_loss /= total
            self.val_loss.append(val_loss)
            try:
                self.val_kl_loss.append(val_epoch_kl_loss.item())
            except:
                self.val_kl_loss.append(val_epoch_kl_loss)
            
            #Exportación de activaciones como numpy
            if histogram_dataset is not None:
                self.export_activations(histogram_dataset,epoch)

            epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_epoch_kl_loss=f"{train_epoch_kl_loss:.4f}",
            train_acc=f"{train_acc*100:.2f}%",
            val_loss=f"{val_loss:.4f}",
            val_epoch_kl_loss=f"{val_epoch_kl_loss:.4f}",
            val_acc=f"{val_acc*100:.2f}%")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_dict(epoch)

        print("Entrenamiento Terminado")
        self.save_losses()
        
    def evaluate(self,dataset):
        """
        Evalúa el modelo en un dataset dado.

        Args:
            dataset (Dataset): Dataset a evaluar.

        Returns:
            dict: Diccionario con etiquetas reales, predichas y métricas (accuracy, losses).
        """
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        self.model.eval()
        real_labels = []
        pred_labels = []
        with torch.no_grad():
            total_bce_loss = 0.0   
            total_kl_loss = 0.0
            for x, y in tqdm(dataloader, desc="Testing", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                output, kl_loss = self.model(x)
                bce_loss = self.criterion(output, y)
                total_bce_loss += bce_loss.item()
                if type(kl_loss)==int:
                    total_kl_loss += self.beta*kl_loss
                else:
                    total_kl_loss += self.beta*kl_loss.item()
                real_labels.append(y.cpu().item()); pred_labels.append(output.argmax(dim=1).cpu().item())

            total_kl_loss/=len(dataset)
            total_bce_loss/=len(dataset)
        real_labels=np.array(real_labels); pred_labels=np.array(pred_labels)
        acc = 100*np.sum(real_labels==pred_labels)/len(dataset)
        dict_results = {"real_labels":real_labels,
                        "pred_labels":pred_labels,
                        "accuracy":acc,
                        "kl_loss": total_kl_loss,
                        "bce_loss": total_bce_loss}

        return dict_results

    def export_activations(self,dataset,epoch:int):
        """
        Exporta las activaciones de la capa oculta a archivos .npy.

        Args:
            dataset (Dataset): Dataset para pasar por el modelo.
            epoch (int): Número de época actual.
        """
        original_out_mlp = self.model.out_mlp
        self.model.out_mlp= nn.Identity()
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        self.model.eval()
        outputs=[]
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Exporting ReLU Histogram", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                output, kl_loss = self.model(x)
                outputs.append(output.cpu().numpy())

        activations = np.concatenate(outputs).reshape(-1)
        #Se restaura modelo
        self.model.out_mlp = original_out_mlp
        #Se guardan los npy
        npy_dir = os.path.join(self.save_plots_dir, "activations_npy")
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, f"epoch_{epoch:03d}.npy"), activations)


    def save_dict(self,epoch):
        """
        Guarda el estado del modelo, optimizador y scheduler.

        Args:
            epoch (int): Época actual.
        """
        print(f"Saving checkpoint to {self.save_model_dir}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, os.path.join(self.save_model_dir,f"{self.seed}.pth"))

    def load_model(self,path):
        """
        Carga los pesos del modelo desde un archivo.

        Args:
            path (str): Ruta al archivo .pth.
        """
        checkpoint = torch.load(path,map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def save_losses(self):
        """
        Guarda las pérdidas de entrenamiento y validación usando la función utilitaria `save_losses`.
        """
        losses_dict={"train_losses":self.train_loss,"val_losses":self.val_loss, "train_kl_losses":self.train_kl_loss, "val_kl_losses":self.val_kl_loss}
        save_losses(losses_dict=losses_dict,
                    losses_dir=self.save_losses_dir,
                    plots_dir=self.save_plots_dir,
                    seed=self.seed)
    


    
class EngineMLPTest:
    """
    Clase simplificada para probar un modelo ya entrenado.
    """
    def __init__(self, mlp_config,train_config,device):
        """
        Inicializa el motor de prueba.

        Args:
            mlp_config (dict): Configuración del modelo.
            train_config (dict): Configuración de entrenamiento (para criterio y beta).
            device (torch.device): Dispositivo a utilizar.
        """
        self.model = FullyConnectedPaper(**mlp_config).to(device)
        self.device = device
        self.criterion = set_criterion(train_config["criterion"])
        self.beta = train_config["beta"]

    def load_model(self,path):
        """
        Carga los pesos del modelo.

        Args:
            path (str): Ruta al archivo .pth.
        """
        checkpoint = torch.load(path,map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded")
    
    def test(self,test_dataset):
        """
        Evalúa el modelo en el dataset de prueba.

        Args:
            test_dataset (Dataset): Dataset de prueba.

        Returns:
            dict: Diccionario con resultados y métricas.
        """
        dataloader = DataLoader(test_dataset,batch_size=1,shuffle=False)
        self.model.eval()
        real_labels = []
        pred_labels = []
        total_bce_loss = 0.0   
        total_kl_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Testing", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                output,kl_loss = self.model(x)
                real_labels.append(y.cpu().item()); pred_labels.append(output.argmax(dim=1).cpu().item())
                total_bce_loss += self.criterion(output, y).item()
                if type(kl_loss)==int:
                    total_kl_loss += self.beta*kl_loss
                else:
                    total_kl_loss += self.beta*kl_loss.item()
        total_bce_loss /= len(test_dataset)
        total_kl_loss /= len(test_dataset)
        real_labels=np.array(real_labels); pred_labels=np.array(pred_labels)
        acc = 100*np.sum(real_labels==pred_labels)/len(test_dataset)
        dict_results = {"real_labels":real_labels,
                        "pred_labels":pred_labels,
                        "accuracy":acc,
                        "bce_loss":total_bce_loss,
                        "kl_loss":total_kl_loss}
        return dict_results