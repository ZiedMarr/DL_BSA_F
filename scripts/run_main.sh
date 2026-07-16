#!/bin/bash
#SBATCH --job-name=ecg_cnn_100ep
#SBATCH --partition=gpu-teaching-2d
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out
#SBATCH --chdir=/home/bsa01/DL_BSA_F

apptainer run --nv ~/containers/bsa05_cont_3.sif python -u main.py --model cnn_vit --epochs 100