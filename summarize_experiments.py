import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(prog='Summary Experiments')
parser.add_argument('--experiment', type=str) #Nombre de la carpeta del experimento en experiments
parser.add_argument('--criterium', type=str, default="accuracy") #Criterio para elegir el mejor experimento: "val_loss" o "val_acc"
args = parser.parse_args()
folder_experiments = args.experiment
criterium = args.criterium
assert criterium in ["accuracy","kl_loss","bce_loss"], "Criterium debe ser 'accuracy', 'kl_loss' o 'bce_loss'"
results_folder = os.path.join("experiments",folder_experiments,"results")
experiment_names = os.listdir(results_folder)
summary_list = []
for exp_name in experiment_names:
    exp_path = os.path.join(results_folder,exp_name)
    csv_summary = pd.read_csv(os.path.join(exp_path,"summary_test.csv"))
    seeds = csv_summary["seed"].values
    mean_acc = np.mean(csv_summary["accuracy"].values)
    std_acc = np.std(csv_summary["accuracy"].values)
    mean_kl = np.mean(csv_summary["kl_loss"].values)
    std_kl = np.std(csv_summary["kl_loss"].values)
    mean_bce = np.mean(csv_summary["bce_loss"].values)
    std_bce = np.std(csv_summary["bce_loss"].values)
    mean_total_loss = np.mean(csv_summary["kl_loss"].values + csv_summary["bce_loss"].values)
    std_total_loss = np.std(csv_summary["kl_loss"].values + csv_summary["bce_loss"].values)
    best_seed = csv_summary.loc[csv_summary[criterium].idxmax()]["seed"] if args.criterium =="accuracy" else csv_summary.loc[csv_summary[criterium].idxmin()]["seed"]
    csv_summary.set_index("seed",inplace=True)
    best_seed_acc = csv_summary.loc[best_seed]["accuracy"]
    best_seed_kl = csv_summary.loc[best_seed]["kl_loss"]
    best_seed_bce = csv_summary.loc[best_seed]["bce_loss"]
    best_seed_total_loss = best_seed_kl + best_seed_bce
    summary_list.append({"folder":folder_experiments,
                        "experiment":exp_name,
                        "mean_accuracy":mean_acc,
                        "std_accuracy":std_acc,
                        "mean_kl_loss":mean_kl,
                        "std_kl_loss":std_kl,
                        "mean_bce_loss":mean_bce,
                        "std_bce_loss":std_bce,
                        "mean_total_loss":mean_total_loss,
                        "std_total_loss":std_total_loss,
                        "best_seed":best_seed,
                        "best_seed_accuracy":best_seed_acc,
                        "best_seed_kl_loss":best_seed_kl,
                        "best_seed_bce_loss":best_seed_bce,
                        "best_seed_total_loss":best_seed_total_loss})
df_summary = pd.DataFrame(summary_list)
summary_path = os.path.join("experiments",folder_experiments,"summary_experiments.csv")
df_summary.to_csv(summary_path,index=False)
print(f"Summary saved in {summary_path}")




