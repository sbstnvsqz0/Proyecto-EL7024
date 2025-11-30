import torch.nn as nn
import torch
from typing import List, Dict, Any
import math

class InformationDropout(nn.Module): #Utiliza var estática
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
    
class InformationDropoutMLP(nn.Module): #Utiliza var dependiente de la entrada
    def __init__(self,input_features: int,output_features:int, initial_logvar: float = 0.0):
        super().__init__()
        assert initial_logvar <=0, "initial_logvar debe ser menor o igual a 0"
        self.logvar_predictor = nn.Linear(input_features, output_features)
        self.max_var = 0.7 #Paper: To avoid this problem, we constraint alpha(x) < 0.7 
        self.log_max_var = math.log(self.max_var)   
        self.log_min_var = math.log(1e-6)              
        
    
        init_logvar = max(min(initial_logvar, self.log_max_var), self.log_min_var)
        nn.init.zeros_(self.logvar_predictor.weight) #Se inicializan pesos en 0 
        nn.init.constant_(self.logvar_predictor.bias, init_logvar) #Se inicializa bias para que inicialmente var=exp(initial_logvar.biases)

    def forward(self,z,x):
        # Z es el mapa de caracteristicas intermedio, X es la entrada completa con el cual se predice el ruido
        logvar = self.logvar_predictor(x)
        logvar = torch.clamp(logvar, self.log_min_var, self.log_max_var) #clamping to avoid numerical instability
        var = torch.exp(logvar)

        kl_loss = torch.mean(torch.sum(-torch.log(var),dim=1)) #mean over batch, sum over features
        if self.training:
            noise = torch.randn_like(z) * torch.sqrt(var) #N(0,var)
            output = z*(1+noise)    #N(x,var)

        else: 
            output=z    #No se aplica ruido en evaluación

        return output, kl_loss
        

class MLPBlock(nn.Module):
    def __init__(self,
                 in_dim:int,
                 out_dim:int,
                 dropout:Dict[str, Any]):
        super().__init__()
        assert dropout["type"] in ["standard","information_static","information"]
        self.dropout_type = dropout["type"]

        if self.dropout_type == "standard":
            dropout_layer = nn.Dropout(dropout["p"])
        elif self.dropout_type == "information_static":
            dropout_layer = InformationDropout(input_features=out_dim, initial_logvar=dropout["initial_logvar"])
        elif self.dropout_type == "information": #information
            dropout_layer = InformationDropoutMLP(input_features=in_dim,output_features=out_dim, initial_logvar=dropout["initial_logvar"])
        self.mlp_layer = nn.Linear(in_dim,out_dim)
        self.dropout_layer = dropout_layer
        self.activation = nn.ReLU()

    def forward(self,x):
        z = self.mlp_layer(x)
        z = self.activation(z)
        if self.dropout_type == "standard":
            z = self.dropout_layer(z)
            kl_loss = 0
        elif self.dropout_type == "information_static":
            z, kl_loss = self.dropout_layer(z)
            
        elif self.dropout_type == "information":
            z, kl_loss = self.dropout_layer(z,x)
        return z, kl_loss

        

class FullyConnectedPaper(nn.Module):
    def __init__(self, 
                 in_dim:int,
                 hidden_dim:int,
                 out_dim:int,
                 dropout:Dict[str, Any]):
        super().__init__()

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
    
        

