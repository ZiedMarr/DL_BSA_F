#!/bin/bash
#SBATCH --job-name=ecg_training
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out
#SBATCH --chdir=~/repo/DL_BSA_F

apptainer run --nv ~/containers/python_container.sif python main.py