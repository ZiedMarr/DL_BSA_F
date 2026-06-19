#!/bin/bash
#SBATCH --job-name=ecg_training
#SBATCH --partition=gpu-teaching-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out
#SBATCH --chdir=~/repo/DL_BSA_F
source ~/.wandb_key
export WANDB_PROJECT=ecg-classification
export WANDB_ENTITY=zied-marrakchi2001

apptainer run --nv ~/containers/python_bsa05_container.sif python main.py