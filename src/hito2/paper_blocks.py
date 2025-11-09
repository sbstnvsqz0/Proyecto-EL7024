import torch.nn as nn
import torch
from typing import List, Dict, Any

class InformationDropout(nn.Module):
    def __init__(self,input_features: int,initial_logvar: float = 0.0):
        super().__init__()
        self.logvar = nn.Parameter(torch.full((input_features,), initial_logvar,dtype=torch.float32))
        self.max_var = 0.7 #Paper: To avoid this problem, we constraint alpha(x) < 0.7
    def forward(self,x):
        var = torch.exp(self.logvar)
        var = torch.clamp(var, 1e-6, self.max_var) #clamping to avoid numerical instability
        if self.training:
            noise = torch.randn_like(x) * torch.sqrt(var) #N(0,var)
            output = x*(1+noise)    #N(x,var)
            kl_loss = -torch.log(var).sum() #sum para las caracteristicas
        else: 
            output=x    #No se aplica ruido en evaluación
            kl_loss = -torch.log(var).sum() 
            
        return output, kl_loss
        

class MLPBlock(nn.Module):
    def __init__(self,
                 in_dim:int,
                 out_dim:int,
                 dropout:Dict[str, Any]):
        super().__init__()
        assert dropout["type"] in ["standard","information"]
        self.dropout_type = dropout["type"]

        if self.dropout_type == "standard":
            dropout_layer = nn.Dropout(dropout["p"])
        else:
            dropout_layer = InformationDropout(input_features=out_dim, initial_logvar=dropout["initial_logvar"])
        
        self.mlp_layer = nn.Linear(in_dim,out_dim)
        self.dropout_layer = dropout_layer
        self.activation = nn.ReLU()

    def forward(self,x):
        x = self.mlp_layer(x)
        x = self.activation(x)
        if self.dropout_type == "standard":
            x = self.dropout_layer(x)
            kl_loss = 0
        else:
            x, kl_loss = self.dropout_layer(x)
        return x, kl_loss

        

class FullyConnectedPaper(nn.Module):
    def __init__(self, 
                 in_dim:int,
                 hidden_dim:int,
                 out_dim:int,
                 dropout:Dict[str, Any]):
        super().__init__()
        assert dropout["type"] in ["standard","information"]

        self.first_mlp = MLPBlock(in_dim=in_dim,
                                  out_dim=hidden_dim,
                                  dropout=dropout)
        #self.second_mlp = MLPBlock(in_dim=hidden_dim,
        #                          out_dim=hidden_dim,
        #                          dropout=dropout)
        #self.third_mlp = MLPBlock(in_dim=hidden_dim,
        #                          out_dim=hidden_dim,
        #                          dropout=dropout)
        self.out_mlp = nn.Linear(hidden_dim,out_dim)
        

    def forward(self, x: torch.Tensor):
        x = nn.Flatten()(x) #Aplana entrada para asegurar que entra un vector
        x1,kl_loss1 = self.first_mlp(x)
        #x2,kl_loss2 = self.second_mlp(x1)
        #x3,kl_loss3 = self.third_mlp(x2)
        logits = self.out_mlp(x1)
        #kl_loss = kl_loss1 + kl_loss2 + kl_loss3
        kl_loss = kl_loss1

        return logits,kl_loss
    
        

