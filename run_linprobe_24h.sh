#!/bin/bash
#SBATCH --job-name=linprobe_24h
#SBATCH --account=cis251356-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/linprobe_24h_%j.log
#SBATCH --error=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/linprobe_24h_%j.err

module load conda/2025.02
eval "$(conda shell.bash hook)"
conda activate surya

cd /anvil/scratch/x-kyang10/Surya/downstream_examples/ar_segmentation
mkdir -p logs checkpoints_linprobe_24h

export PYTHONPATH=/anvil/scratch/x-kyang10/Surya:$PYTHONPATH
export MASTER_ADDR=localhost
export MASTER_PORT=29503
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0

python -u finetune.py --gpu --wandb --config_path ./config_linear_probe_24h.yaml 2>&1
echo "=== Exit code: $? ==="
