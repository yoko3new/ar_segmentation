#!/bin/bash
#SBATCH --job-name=test_hmi_only
#SBATCH --account=cis251356-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_hmi_only_%j.log
#SBATCH --error=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_hmi_only_%j.err

module load conda/2025.02
eval "$(conda shell.bash hook)"
conda activate surya

cd /anvil/scratch/x-kyang10/Surya/downstream_examples/ar_segmentation
export PYTHONPATH=/anvil/scratch/x-kyang10/Surya:$PYTHONPATH
mkdir -p test_results test_preds/hmi_only

python test.py \
    --config_path ./config_hmi_only.yaml \
    --checkpoint_path ./checkpoints_hmi_only/epoch_14.pth \
    --nc_dir /anvil/scratch/x-kyang10/test_nc \
    --hour_filter 0 \
    --output_csv ./test_results/hmi_only.csv \
    --save_preds_dir ./test_preds/hmi_only 2>&1
echo "=== Exit code: $? ==="
