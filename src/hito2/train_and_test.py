from torchvision.datasets import MNIST
from.engine import EngineMLP
import argparse
import os
import yaml
from torchvision import transforms

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER,exist_ok=True)

parser = argparse.ArgumentParser(prog='Train and Test')
parser.add_argument('--experiment', type=str) #.yaml

args = parser.parse_args()
exp_file = args.experiment
assert exp_file.endswith(".yaml"), "Se debe ingresar un archivo .yaml"
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

#Se lee el archivo .yaml con las configuraciones
with open(exp_file, 'r') as file:
    exp_config = yaml.safe_load(file)

seeds = exp_config["seeds"]
preprocessing_config = exp_config["preprocessing_config"]
model_config = exp_config["model_config"]
train_config = exp_config["train_config"]

#Inicialización de Datasets
transforms = transforms.Compose([
    transforms.Resize(preprocessing_config["size"]),
    transforms.ToTensor(), 
    transforms.Normalize((preprocessing_config["mean"],), (preprocessing_config["std"],))
])
model_config["in_dim"] = preprocessing_config["size"]**2 #Se configura la entrada del modelo según el resize
if not os.path.isdir(os.path.join(DATA_FOLDER,"mnist")):
    print("Descargando dataset MNIST en carpeta data")
train_dataset = MNIST(DATA_FOLDER, train=True, download=True, transform=transforms)
test_dataset = MNIST(DATA_FOLDER, train=False, download=True, transform=transforms)

for seed in seeds:
    engine = EngineMLP(seed=seed, 
                    save_model_dir=model_save_dir, 
                    mlp_config=model_config, 
                    train_config=train_config)
    engine.train(train_dataset=train_dataset,
                val_dataset=test_dataset)


