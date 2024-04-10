#!/bin/bash
#SBATCH --job-name="manet-run"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bbmr-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 47:59:00

pwd
conda deactivate
module load anaconda3_gpu/23.9.0
source activate control
nvidia-smi
python train_manet.py