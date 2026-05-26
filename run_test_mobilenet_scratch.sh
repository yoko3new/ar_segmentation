#!/bin/bash
#SBATCH --job-name=test_mobilenet_scratch
#SBATCH --account=cis251356-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_mobilenet_scratch_%j.log
#SBATCH --error=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_mobilenet_scratch_%j.err

module load conda/2025.02
eval "$(conda shell.bash hook)"
conda activate surya

cd /anvil/scratch/x-kyang10/Surya/downstream_examples/ar_segmentation
export PYTHONPATH=/anvil/scratch/x-kyang10/Surya:$PYTHONPATH
mkdir -p test_results test_preds/mobilenet_scratch

python test.py \
    --config_path ./config_mobilenet_scratch.yaml \
    --checkpoint_path ./checkpoints_mobilenet_scratch/epoch_13.pth \
    --nc_dir /anvil/scratch/x-kyang10/test_nc \
    --hour_filter 0 \
    --output_csv ./test_results/mobilenet_scratch.csv \
    --save_preds_dir ./test_preds/mobilenet_scratch 2>&1
echo "=== Exit code: $? ==="
