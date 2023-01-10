#! /bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=at_dts
#SBATCH --mail-type=ALL
#SBATCH --time 1-00:00:00
#SBATCH --mem=20000
#SBATCH --output=R-%x.%j.out

cd $HOME/23_ExAutoML/2022AutoML_Causality
module load python/3.8.6
source venv/bin/activate

cd $HOME/23_ExAutoML/2022AutoML_Causality/main_py
python 01_autocausality_DTs.py ${1}