#!/bin/bash
#SBATCH --job-name=ecg_training
#SBATCH --partition=gpu-teaching-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out
#SBATCH --chdir=/home/bsa05/repo/DL_BSA_F
source ~/.wandb_key
export WANDB_PROJECT=ecg-classification

apptainer run --nv ~/containers/bsa05_cont.sif python main.py