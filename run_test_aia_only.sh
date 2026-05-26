#!/bin/bash
#SBATCH --job-name=test_aia_only
#SBATCH --account=cis251356-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_aia_only_%j.log
#SBATCH --error=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_aia_only_%j.err

module load conda/2025.02
eval "$(conda shell.bash hook)"
conda activate surya

cd /anvil/scratch/x-kyang10/Surya/downstream_examples/ar_segmentation
export PYTHONPATH=/anvil/scratch/x-kyang10/Surya:$PYTHONPATH
mkdir -p test_results test_preds/aia_only

python test.py \
    --config_path ./config_aia_only.yaml \
    --checkpoint_path ./checkpoints_aia_only/epoch_14.pth \
    --nc_dir /anvil/scratch/x-kyang10/test_nc \
    --hour_filter 0 \
    --output_csv ./test_results/aia_only.csv \
    --save_preds_dir ./test_preds/aia_only 2>&1
echo "=== Exit code: $? ==="
