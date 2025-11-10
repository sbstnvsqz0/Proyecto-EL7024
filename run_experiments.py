import os
import subprocess
import time
import yaml
experiment_name = "hito2_full"
yaml_dir = os.path.join("experiments",f"{experiment_name}")
yaml_files = [f for f in os.listdir(yaml_dir) if f.endswith('.yaml')]
for yaml_file in yaml_files:
    try:
        yaml_path = os.path.join(yaml_dir,yaml_file)
        with open(yaml_path, 'r') as file:
            exp_config = yaml.safe_load(file)

        seeds = exp_config["seeds"]
        for seed in seeds:
            command = f"python -m src.hito2.train_and_test --experiment={yaml_path} --seed={seed}"
            print(f'Running command: {command}')
            time.sleep(2)
            result1 = subprocess.run(command, shell=True)

    except Exception as e:
        print(f"Error with experiment {yaml_file}: {e}")

command2 = f"python summarize_experiments.py --experiment={experiment_name} --criterium=accuracy"
print(f'Running command: {command2}')
time.sleep(2)
result2 = subprocess.run(command2, shell=True)