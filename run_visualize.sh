#!/bin/bash
#SBATCH --job-name=ar_visualize
#SBATCH --account=cis251356-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/visualize_%j.log
#SBATCH --error=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/visualize_%j.err

module load conda/2025.02
eval "$(conda shell.bash hook)"
conda activate surya

cd /anvil/scratch/x-kyang10/Surya/downstream_examples/ar_segmentation
mkdir -p visualization_results

export PYTHONPATH=/anvil/scratch/x-kyang10/Surya:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python visualize.py \
    --config_path ./config_visualize.yaml \
    --checkpoint_path ./checkpoints_resume/epoch_20.pth \
    --output_dir ./visualization_results \
    --num_samples 3 2>&1
echo "=== Exit code: $? ==="
