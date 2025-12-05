from torchvision.datasets import MNIST
from.engine import EngineMLP
import argparse
import os
import yaml
from torchvision import transforms
from src.utils import results_per_seed, save_losses, create_folders, create_stratified_subset
from .noises_transforms import choose_noise, DatasetLabelNoise
DATA_FOLDER = "data"

def main():
    
    os.makedirs(DATA_FOLDER,exist_ok=True)

    parser = argparse.ArgumentParser(prog='Train and Test')
    parser.add_argument('--experiment', type=str) #.yaml
    parser.add_argument('--seed',type=int)  #seed
    parser.add_argument('--train',type=str,default="True")
    parser.add_argument('--get_histograms',type=str,default="False")

    args = parser.parse_args()
    exp_file = args.experiment
    seed = args.seed
    train = args.train
    get_histograms = args.get_histograms
    train = False if str(train).lower() == "false" else True
    get_histograms = False if str(get_histograms).lower() == "false" else True

    assert exp_file.endswith(".yaml"), "Se debe ingresar un archivo .yaml"

    #Se crean carpetas
    folder_experiments,model_save_dir,losses_save_dir,preds_save_dir,plots_save_dir = create_folders(exp_file)

    #Se lee el archivo .yaml con las configuraciones
    with open(exp_file, 'r') as file:
        exp_config = yaml.safe_load(file)

    n_seeds = len(exp_config["seeds"])
    preprocessing_config = exp_config["preprocessing_config"]
    model_config = exp_config["model_config"]
    train_config = exp_config["train_config"]

    if "noise" in preprocessing_config.keys():
        #Inicialización de Datasets
        if preprocessing_config["noise"]["type"]=="label":
            preprocessing = transforms.Compose([
                transforms.Resize(preprocessing_config["size"]),
                transforms.ToTensor()])
        else:
            preprocessing = transforms.Compose([
                transforms.Resize(preprocessing_config["size"]),
                transforms.ToTensor(),
                choose_noise(noise_type=preprocessing_config["noise"]["type"], param=preprocessing_config["noise"]["param"])
            ])
    else:
        preprocessing = transforms.Compose([
            transforms.Resize(preprocessing_config["size"]),
            transforms.ToTensor()])

    model_config["in_dim"] = preprocessing_config["size"]**2 #Se configura la entrada del modelo según el resize

    if not os.path.isdir(os.path.join(DATA_FOLDER,"mnist")):
        print("Descargando dataset MNIST en carpeta data")
    
    train_dataset = MNIST(DATA_FOLDER, train=True, download=True, transform=preprocessing)
    try:
        if preprocessing_config["noise"]["type"]=="label":
            train_dataset = DatasetLabelNoise(dataset_original=train_dataset,p=preprocessing_config["noise"]["param"])
    except:
        pass

    test_dataset = MNIST(DATA_FOLDER, train=False, download=True, transform=preprocessing)

    #Dataset para creación de histogramas de activación (10% de test estratificado por label)
    test_dataset_histograms=None
    if get_histograms:
        test_dataset_histograms = create_stratified_subset(seed=seed, dataset = test_dataset, subset_size=0.1)


    engine = EngineMLP(seed=seed, 
                    save_model_dir=model_save_dir,
                    save_losses_dir= losses_save_dir,
                    save_plots_dir = plots_save_dir,
                    mlp_config=model_config, 
                    train_config=train_config)
    if train:
        engine.train(train_dataset=train_dataset,
                    val_dataset=test_dataset,
                    histogram_dataset=test_dataset_histograms)

    engine.load_model(os.path.join(model_save_dir,f"{seed}.pth"))
    test_results = engine.evaluate(dataset = test_dataset)
    # Genera csv con etiqueta real y predicha; matriz de confusión
    results_per_seed(results_dict=test_results,
                    preds_save_dir=preds_save_dir,
                    plots_save_dir=plots_save_dir,
                    summary_dir=folder_experiments,
                    seed=seed,
                    n_seeds=n_seeds,
                    set_name="test")
    
    
if __name__=="__main__":
    main()    




