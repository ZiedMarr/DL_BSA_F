#!/bin/bash
# submit_array.sbatch
#
# Workflow before submitting:
#   1. python generate_grid.py          # regenerate hparam_grid.json
#   2. Update --array below to match the printed range
#   3. sbatch scripts/submit_array.sbatch

#SBATCH --job-name=ecg_sweep
#SBATCH --partition=gpu-teaching-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=02:00:00
#SBATCH --array=0-17          # update this after regenerating hparam_grid.json
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --chdir=/home/bsa05/repo/DL_BSA_F

mkdir -p logs

source ~/.wandb_key
export WANDB_PROJECT=ecg-classification
export WANDB_ENTITY=zied-marrakchi2001-technical-university-of-berlin

apptainer exec --nv ~/containers/bsa05_cont_3.sif \
    python launch_task.py $SLURM_ARRAY_TASK_ID
