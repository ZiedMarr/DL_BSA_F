#!/bin/bash
#SBATCH --job-name=ecg_weighted_reg5
#SBATCH --partition=gpu-teaching-2d
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/weighted-reg1e5-%j.out
#SBATCH --chdir=/home/bsa01/DL_BSA_F

apptainer run --nv /home/bsa01/containers/python_container.sif python -u main.py --model cnn1d --epochs 100 --class-weights --learning-rate 1e-3 --weight-decay 1e-5 --protocol group_kfold --experiment-name weighted_reg1e5
apptainer run --nv /home/bsa01/containers/python_container.sif python -u main.py --model resnet --epochs 100 --class-weights --learning-rate 1e-3 --weight-decay 1e-5 --protocol group_kfold --experiment-name weighted_reg1e5
