#!/bin/bash
#SBATCH --job-name=test_lora_bce
#SBATCH --account=cis251356-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_lora_bce_%j.log
#SBATCH --error=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_lora_bce_%j.err

module load conda/2025.02
eval "$(conda shell.bash hook)"
conda activate surya

cd /anvil/scratch/x-kyang10/Surya/downstream_examples/ar_segmentation
export PYTHONPATH=/anvil/scratch/x-kyang10/Surya:$PYTHONPATH
mkdir -p test_results test_preds/lora_bce

python test.py \
    --config_path ./config.yaml \
    --checkpoint_path ~/ar_seg_checkpoints/spectformer/epoch_31.pth \
    --nc_dir /anvil/scratch/x-kyang10/test_nc \
    --hour_filter 0 \
    --output_csv ./test_results/lora_bce.csv \
    --save_preds_dir ./test_preds/lora_bce 2>&1
echo "=== Exit code: $? ==="
