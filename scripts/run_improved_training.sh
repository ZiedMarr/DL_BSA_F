#!/bin/bash
#SBATCH --job-name=ecg_improved
#SBATCH --partition=gpu-teaching-2d
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/improved-%j.out
#SBATCH --chdir=/home/bsa01/DL_BSA_F

apptainer run --nv /home/bsa01/containers/python_container.sif python -u main.py --model cnn1d --epochs 100 --class-weights --class-weight-mode sqrt --learning-rate 5e-4 --weight-decay 1e-4 --threshold-tuning --protocol group_kfold --experiment-name improved
apptainer run --nv /home/bsa01/containers/python_container.sif python -u main.py --model resnet --epochs 100 --class-weights --class-weight-mode sqrt --learning-rate 5e-4 --weight-decay 1e-4 --threshold-tuning --protocol group_kfold --experiment-name improved
