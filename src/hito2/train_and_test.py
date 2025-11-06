from torchvision.datasets import MNIST
from.engine import EngineMLP
import argparse
import os
import yaml
from torchvision import transforms
from src.utils import results_per_seed, save_losses

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER,exist_ok=True)

parser = argparse.ArgumentParser(prog='Train and Test')
parser.add_argument('--experiment', type=str) #.yaml
parser.add_argument('--seed',type=int)  #seed

args = parser.parse_args()
exp_file = args.experiment
seed = args.seed

assert exp_file.endswith(".yaml"), "Se debe ingresar un archivo .yaml"
#TODO: Pasar a funcion la creación de dirs
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

#Se lee el archivo .yaml con las configuraciones
with open(exp_file, 'r') as file:
    exp_config = yaml.safe_load(file)

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

if __name__=="__main__":
    engine = EngineMLP(seed=seed, 
                    save_model_dir=model_save_dir,
                    save_losses_dir= losses_save_dir,
                    save_plots_dir = plots_save_dir,
                    mlp_config=model_config, 
                    train_config=train_config)

    engine.train(train_dataset=train_dataset,
                val_dataset=test_dataset)

    engine.load_model(os.path.join(model_save_dir,f"{seed}.pth"))
    
    test_results = engine.evaluate(dataset = test_dataset)
    # Genera csv con etiqueta real y predicha; matriz de confusión
    results_per_seed(results_dict=test_results,
                    preds_save_dir=preds_save_dir,
                    plots_save_dir=plots_save_dir,
                    seed=seed,
                    set_name="test")
    
    
    




