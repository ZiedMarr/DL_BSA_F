#!/bin/bash
#SBATCH --job-name=ecg_m4
#SBATCH --partition=gpu-teaching-2d
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/m4-%j.out
#SBATCH --chdir=/home/bsa01/DL_BSA_F

apptainer run --nv /home/bsa01/containers/python_container.sif python -u main.py --model cnn1d --epochs 200 --class-weights --class-weight-mode sqrt --learning-rate 5e-5 --weight-decay 5e-4 --optimizer adamw --scheduler warmup_cosine --warmup-epochs 5 --augment --experiment-name m4
apptainer run --nv /home/bsa01/containers/python_container.sif python -u main.py --model resnet --epochs 200 --class-weights --class-weight-mode sqrt --learning-rate 5e-5 --weight-decay 5e-4 --optimizer adamw --scheduler warmup_cosine --warmup-epochs 5 --augment --experiment-name m4
