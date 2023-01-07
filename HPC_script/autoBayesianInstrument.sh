#!/bin/bash
#SBATCH -J AutoBayesian
#SBATCH -N 1
#SBATCH --cpus-per-task=10
#SBATCH -t 24:45:00
#SBATCH --mem=128G
#SBATCH --partition=amd
module load python/3.8.6
cd $HOME/AutoML
python autocausality_AutoBayesian_addInstrument.py
