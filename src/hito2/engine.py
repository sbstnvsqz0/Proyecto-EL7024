from .paper_blocks import FullyConnectedPaper
from typing import Dict, Any
import torch
from tqdm import tqdm
from src.utils import (set_seed, device_auto,set_criterion,set_optimizer)
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

class EngineMLP:
    def __init__(self, seed, save_model_dir, mlp_config: Dict[str, Any], train_config: Dict[str, Any]):
        self.seed = seed
        set_seed(self.seed)
        self.device = device_auto()
        self.save_model_dir = save_model_dir
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
            train_acc=f"{train_acc:.2f}%",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.2f}%")
            if epoch<self.step_size+1:    #Solo una actualización del lr como en el paper
                self.scheduler.step()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_dict(epoch)
        
    def save_dict(self,epoch):
        print(f"Saving checkpoint to {self.save_model_dir}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, os.path.join(self.save_model_dir,f"{self.seed}.pth"))

    def return_losses(self):
        return {"train":self.train_loss, "val":self.val_loss}


    
