import torch
from torch.utils.data import Dataset
import random
import numpy as np

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
        self.noise_mapping = {9:[7,4],4:9, 7:9, 3:[5,8],5:[3,8],8:[3,5],1:2,2:1,6:0,0:6} #Basado en cercanías mostradas en UMAP 
        if isinstance(self.dataset_original_y, torch.Tensor):
            self.dataset_original_y = self.dataset_original_y.numpy()
        else:
            self.dataset_original_y = np.array(self.dataset_original_y)

        rng = np.random.default_rng(42)

        self.noise_mask = [random.random() < self.p for _ in range(len(self.dataset_original_y))]
        self.num_classes = 10 #Hardcodeada
        self.new_labels = self.dataset_original_y.copy()
        for source_c, target_c in self.noise_mapping.items():
            targets = target_c if isinstance(target_c, list) else [target_c]
            indices = np.where(self.dataset_original_y == source_c)[0]
            total_flip_count = int(len(indices) * p)
            if total_flip_count > 0:
                rng.shuffle(indices)
                indices_to_corrupt = indices[:total_flip_count]
                target_chunks = np.array_split(indices_to_corrupt, len(targets))
                for chunk_indices, target_label in zip(target_chunks, targets):
                    self.new_labels[chunk_indices] = target_label

        self.new_labels = self.new_labels.tolist()
        actual_noise = np.mean(np.array(self.new_labels) != self.dataset_original_y)
        print(f"Global noise: {actual_noise:.4f}")

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

    