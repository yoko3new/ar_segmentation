#!/bin/bash
#SBATCH --job-name=ar_test_p2
#SBATCH --account=cis251356-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_p2_%j.log
#SBATCH --error=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_p2_%j.err

module load conda/2025.02
eval "$(conda shell.bash hook)"
conda activate surya

cd /anvil/scratch/x-kyang10/Surya/downstream_examples/ar_segmentation
export PYTHONPATH=/anvil/scratch/x-kyang10/Surya:$PYTHONPATH
mkdir -p test_results

NC_DIR="/anvil/scratch/x-kyang10/test_nc"

echo "=========================================="
echo "3. LoRA+BCE+Dice 24h - epoch_14"
echo "=========================================="
python test.py \
    --config_path ./config_24h_bce_dice.yaml \
    --checkpoint_path ./checkpoints_24h_bce_dice/epoch_14.pth \
    --nc_dir $NC_DIR --hour_filter 0 \
    --output_csv ./test_results/lora_bce_dice_24h.csv 2>&1

echo "=========================================="
echo "4. Linear Probing 24h - epoch_17"
echo "=========================================="
python test.py \
    --config_path ./config_linear_probe_24h.yaml \
    --checkpoint_path ./checkpoints_linprobe_24h/epoch_17.pth \
    --nc_dir $NC_DIR --hour_filter 0 \
    --output_csv ./test_results/linprobe_24h.csv 2>&1

echo "=========================================="
echo "5. Linear Probing 12h - epoch_1"
echo "=========================================="
python test.py \
    --config_path ./config_linear_probe_12h.yaml \
    --checkpoint_path ./checkpoints_linprobe_12h/epoch_1.pth \
    --nc_dir $NC_DIR --hour_filter 0 12 \
    --output_csv ./test_results/linprobe_12h.csv 2>&1

echo "=========================================="
echo "PART 2 COMPLETE"
echo "=========================================="
