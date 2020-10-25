import subprocess
from os import path
import os

gammas = [0.1, 0.5, 1]
LR = [0.00001, 0.001]
decays = [0.99999999, 0.9999]

job_num = 0

for gamma in gammas:
    for lr in LR:
        for decay in decays:
            name = f'dqn_{job_num}'
            file = open(
                f'/ubc/cs/research/plai-scratch/virtuecc/launch/{name}.sh', "w+")
            file.write('#!/bin/bash\n')
            file.write(f'#SBATCH --job-name={name}\n')
            file.write('#SBATCH --time=10-20:00:00\n')
            file.write('#SBATCH --partition=plai_cpus\n')
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
                f'python3 Q_network.py --gamma={gamma} --LR={lr} --decay={decay}\n')
            file.close()

            command = f'sbatch /ubc/cs/research/plai-scratch/virtuecc/launch/{file.name}'
            subprocess.call(command, shell=True)
            job_num += 1
