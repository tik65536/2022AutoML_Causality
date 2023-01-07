#! /bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=causality_dts
#SBATCH --mail-type=ALL
#SBATCH --time 1-00:00:00
#SBATCH --mem=20000
#SBATCH --output=R-%x.%j.out

module load python/3.8.6
source venv/bin/activate

python 01_autocausality_DTs.py