import numpy as np
import torch
import random 
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_criterion(criterion:str):
    from torch.nn import CrossEntropyLoss
    assert criterion in ["cross_entropy"]
    return CrossEntropyLoss()

def set_optimizer(optimizer:str):
    from torch.optim import Adam
    assert optimizer in ["adam"]
    return Adam