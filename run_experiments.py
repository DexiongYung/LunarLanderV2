import subprocess
from os import path
import os

gammas = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

for gamma in gammas:
    name = f'exp_gamma_{gamma}'
    file = open(f'/ubc/cs/research/plai-scratch/virtuecc/launch/{name}.sh', "w+")
    file.write('#!/bin/bash\n')
    file.write(f'#SBATCH --job-name={name}\n')
    file.write('#SBATCH --time=10-20:00:00\n')
    file.write('#SBATCH --partition=blackboxml\n')
    file.write('#SBATCH --mem=8G\n')
    file.write('#SBATCH --mail-user=yungdexiong@gmail.com\n')
    file.write('#SBATCH --mail-type=begin\n')
    file.write('#SBATCH --mail-type=begin\n')
    file.write(
        f'#SBATCH --error=logs/{name}.err\n')
    file.write(
        f'#SBATCH --output=logs/{name}.out\n\n')
    file.write(
        'source /ubc/cs/research/plai-scratch/virtuecc/venv/rl/bin/activate\n')
    file.write(
        'cd /ubc/cs/research/plai-scratch/virtuecc/GitHub/LunarLanderV2\n')
    file.write(
        f'python3 experiment.py --gamma={gamma}\n')
    file.close()

    command = f'sbatch {file.name}'
    subprocess.call(command, shell=True)