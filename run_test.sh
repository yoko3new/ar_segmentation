#!/bin/bash
#SBATCH --job-name=ar_seg_test
#SBATCH --account=cis251356-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_%j.log

module load conda/2025.02
conda activate surya

cd ~/Surya/downstream_examples/ar_segmentation
mkdir -p logs checkpoints

torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py --gpu --wandb
