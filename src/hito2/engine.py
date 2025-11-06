from .paper_blocks import FullyConnectedPaper
from typing import Dict, Any
import torch
from tqdm import tqdm
from src.utils import (set_seed, device_auto,set_criterion,set_optimizer, save_losses)
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np

class EngineMLP:
    def __init__(self, seed, save_model_dir, save_losses_dir, save_plots_dir, mlp_config: Dict[str, Any], train_config: Dict[str, Any]):
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
        self.weight_decay = train_config["weight_decay"]
        self.criterion = set_criterion(train_config["criterion"])
        optimizer_class = set_optimizer(train_config["optimizer"])
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.step_size = 30
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=0.1) #
        self.beta = train_config["beta"]
        

        self.best_val_loss = float('inf')
        self.train_loss = []
        self.val_loss = []

    def train(self,train_dataset,val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        epoch_pbar = tqdm(range(self.epochs), desc="Epochs")

        for epoch in epoch_pbar:
            self.model.train()
            train_loss = 0.0
            train_acc = 0
            total = 0
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
            for x, y in train_pbar:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output, kl_loss = self.model(x)
                kl_loss = kl_loss/x.size(0)  
                loss = self.criterion(output, y) + self.beta*kl_loss    
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * x.size(0)
                train_acc += (output.argmax(dim=1) == y).sum().item()
                total += x.size(0) 

            train_loss /= total
            train_acc /= total
            self.train_loss.append(train_loss)

            self.model.eval()
            val_loss = 0.0
            val_acc = 0
            total = 0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc="Validation", leave=False):
                    x, y = x.to(self.device), y.to(self.device)
                    output, kl_loss = self.model(x)
                    loss = self.criterion(output, y) + self.beta*kl_loss/x.size(0)
                    val_loss += loss.item() * x.size(0)
                    val_acc += (output.argmax(dim=1) == y).sum().item()
                    total += x.size(0)
            val_loss /= total
            val_acc /= total
            self.val_loss.append(val_loss)

            epoch_pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc*100:.2f}%",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc*100:.2f}%")
            if epoch<self.step_size+1:    #Solo una actualización del lr como en el paper
                self.scheduler.step()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_dict(epoch)
        print("Entrenamiento Terminado")
        self.save_losses()
        
    def evaluate(self,dataset):
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
        self.model.eval()
        real_labels = []
        pred_labels = []
        with torch.no_grad():
            total_loss = 0.0   
            for x, y in tqdm(dataloader, desc="Testing", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                output, kl_loss = self.model(x)
                loss = self.criterion(output, y) + self.beta*kl_loss/x.size(0)
                total_loss += loss.item() * x.size(0)
                real_labels.append(y.cpu().item()); pred_labels.append(output.argmax(dim=1).cpu().item())

            total_loss/=len(dataset)
        acc = np.sum(real_labels==pred_labels)/len(dataset)
        dict_results = {"real_labels":real_labels,
                        "pred_labels":pred_labels,
                        "accuracy":acc,
                        "loss": total_loss}

        return dict_results


    def save_dict(self,epoch):
        print(f"Saving checkpoint to {self.save_model_dir}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, os.path.join(self.save_model_dir,f"{self.seed}.pth"))

    def load_model(self,path):
        checkpoint = torch.load(path,map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def save_losses(self):
        losses_dict={"train_losses":self.train_loss,"val_losses":self.val_loss}
        save_losses(losses_dict=losses_dict,
                    losses_dir=self.save_losses_dir,
                    plots_dir=self.save_plots_dir,
                    seed=self.seed)
    


    
