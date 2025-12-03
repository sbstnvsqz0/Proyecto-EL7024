import torch
from torch.utils.data import Dataset
import random

class AddGaussianNoise(object):
    def __init__(self, std=0.0):
        assert 1>=std >= 0.0, "std debe ser >= 0.0 y <= 1.0"
        self.std = std
        
    def __call__(self, tensor):
        # Add noise and clamp to ensure valid pixel range
        #Clamp es para que no se salga del rango [0,1]
        tensor = tensor + torch.randn(tensor.size()) * self.std #N(0,self.std)
        tensor = tensor.clamp(min=0.0, max=1.0)
        return tensor

class RandomOcclusion(object):
    def __init__(self, p=0.0):
        assert 1>=p >= 0.0, "p debe ser >= 0.0 y <= 1.0"
        self.p = p
    
    def __call__(self, tensor):
        # Cada punto tiene una probabilidad p de ser ocultado
        mask = torch.rand(tensor.size()) < self.p
        tensor[mask] = 0.0
        return tensor

class ContrastChange(object):
    def __init__(self, factor=1.0):
        assert factor >= 0.0, "factor debe ser >= 0.0"
        self.factor = factor
    
    def __call__(self, tensor):
        # Cambia el contraste de la imagen
        tensor = (tensor-0.5) * self.factor + 0.5
        tensor = tensor.clamp(min=0.0, max=1.0)
        return tensor
    
class DatasetLabelNoise(Dataset):
    def __init__(self, dataset_original:Dataset,p=0.0):
        assert 1>=p >= 0.0, "p debe ser >= 0.0 y <= 1.0"
        self.p = p
        self.dataset_original = dataset_original
        self.dataset_original_y = dataset_original.targets
        if isinstance(self.dataset_original_y, torch.Tensor):
            self.dataset_original_y = self.dataset_original_y.tolist()

        self.noise_mask = [random.random() < self.p for _ in range(len(self.original_labels))]
        
        self.new_labels = []
        for i, y in enumerate(self.original_labels):
            if not self.noise_mask[i]:
                self.new_labels.append(y)
            else:
                noise_label = random.randint(1, self.num_classes - 1)
                new_y = (y + noise_label) % self.num_classes
                self.new_labels.append(new_y)

    def __getitem__(self, idx):
        x,_=self.dataset_original[idx]
        y = self.new_labels[idx]
        return x,y
    
    def __len__(self):
        return len(self.dataset_original)

def choose_noise(noise_type:str, param:float):
    if noise_type == "gaussian":
        return AddGaussianNoise(std=param)
    elif noise_type == "random_occlusion":
        return RandomOcclusion(p=param)
    elif noise_type == "contrast":
        return ContrastChange(factor=param)
    else:
        raise ValueError(f"Noise type {noise_type} not supported")

    