#!/bin/bash
#SBATCH --job-name=get
#SBATCH --time=72:00:00
#SBATCH --partition=gpu_sc
#SBATCH --nodelist=gpu04
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=8
#SBATCH --output=./%j.out
#SBATCH --error=./%j.error
#SBATCH --no-requeue

export CUDA_LAUNCH_BLOCKING=1

python main.py