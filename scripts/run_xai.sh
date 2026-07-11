#!/bin/bash
#SBATCH --job-name=ecg_xai
#SBATCH --partition=gpu-teaching-2h
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/xai-%j.out
#SBATCH --chdir=/home/bsa01/DL_BSA_F

apptainer run --nv /home/bsa01/containers/python_container.sif python -u xai.py --model cnn1d --checkpoint outputs/models/cnn1d_fold0.pt --target-class 6 --lead 1
apptainer run --nv /home/bsa01/containers/python_container.sif python -u xai.py --model resnet --checkpoint outputs/models/resnet_fold0.pt --target-class 6 --lead 1
