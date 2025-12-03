from torchvision.datasets import MNIST
from .engine import EngineMLPTest
import argparse
import os
import yaml
from torchvision import transforms
import torch
from src.hito3.noises_transforms import choose_noise
from src.utils import set_seed, create_folders, results_per_seed

DATA_FOLDER = "data"


def main():
    os.makedirs(DATA_FOLDER,exist_ok=True)

    parser = argparse.ArgumentParser(prog='Train and Test')
    parser.add_argument('--experiment', type=str) #.yaml
    parser.add_argument('--model_seed',type=int)  #seed del modelo entrenado
    parser.add_argument('--noise_seed',default=1234,type=int) #seed para aplicación de ruido 
    parser.add_argument('--noise_type',type=str) #tipo de noise aplicado
    parser.add_argument('--noise_param',type=float) #parametro de noise (depende del noise aplicado)

    args = parser.parse_args()
    exp_file = args.experiment
    model_seed = args.model_seed
    noise_seed = args.noise_seed
    noise_type = args.noise_type
    noise_param = args.noise_param
    assert exp_file.endswith(".yaml"), "Se debe ingresar un archivo .yaml"
    print("Experiment file:", exp_file)
    #Se definen carpetas
    folder_experiments,model_save_dir,losses_save_dir,preds_save_dir,plots_save_dir = create_folders(exp_file)
    #Se lee .YAML
    with open(exp_file, 'r') as file:
        exp_config = yaml.safe_load(file)
    preprocessing_config = exp_config["preprocessing_config"]
    model_config = exp_config["model_config"]
    train_config = exp_config["train_config"]

    set_seed(noise_seed)

    noise_transformation = choose_noise(noise_type=noise_type, param=noise_param)
    preprocessing = transforms.Compose([
        transforms.Resize(preprocessing_config["size"]),
        transforms.ToTensor(),
        noise_transformation
    ])
    model_config["in_dim"] = preprocessing_config["size"]**2 #Se configura la entrada del modelo según el resize
    
    test_set = MNIST(root=DATA_FOLDER, train=False, download=True, transform=preprocessing)
    engine = EngineMLPTest(mlp_config=model_config,train_config=train_config, device="cuda" if torch.cuda.is_available() else "cpu")
    engine.load_model(os.path.join(model_save_dir,f"{model_seed}.pth"))
    test_results = engine.test(test_set)
    results_per_seed(results_dict=test_results,
                    preds_save_dir=preds_save_dir,
                    plots_save_dir=plots_save_dir,
                    summary_dir=folder_experiments,
                    seed=model_seed,
                    n_seeds=3,
                    set_name=f"test_noisy_{noise_type}_{noise_param}")
    
if __name__=="__main__":
    main()

    