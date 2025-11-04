import os
import subprocess
import time

experiment_name = "hito2"
yaml_dir = os.path.join("experiments",f"{experiment_name}")
yaml_files = [f for f in os.listdir(yaml_dir) if f.endswith('.yaml')]

for yaml_file in yaml_files:
    try:
        yaml_path = os.path.join(yaml_dir,yaml_file)
        command = f"python -m src.hito2.train_and_test --experiment={yaml_path}"
        print(f'Running command: {command}')
        time.sleep(2)
        result1 = subprocess.run(command, shell=True)

    except Exception as e:
        print(f"Error with experiment {yaml_file}: {e}")