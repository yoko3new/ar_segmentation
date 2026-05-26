#!/bin/bash
#SBATCH --job-name=ar_test_p3
#SBATCH --account=cis251356-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_p3_%j.log
#SBATCH --error=/home/x-kyang10/Surya/downstream_examples/ar_segmentation/logs/test_p3_%j.err

module load conda/2025.02
eval "$(conda shell.bash hook)"
conda activate surya

cd /anvil/scratch/x-kyang10/Surya/downstream_examples/ar_segmentation
export PYTHONPATH=/anvil/scratch/x-kyang10/Surya:$PYTHONPATH
mkdir -p test_results

NC_DIR="/anvil/scratch/x-kyang10/test_nc"

echo "=========================================="
echo "6. MobileNet ImageNet 24h - epoch_14"
echo "=========================================="
python test.py \
    --config_path ./config_mobilenet.yaml \
    --checkpoint_path ~/ar_seg_checkpoints/mobilenet/epoch_14.pth \
    --nc_dir $NC_DIR --hour_filter 0 \
    --output_csv ./test_results/mobilenet_imagenet_24h.csv 2>&1

echo "=========================================="
echo "7. AIA only 24h - epoch_14"
echo "=========================================="
python test.py \
    --config_path ./config_aia_only.yaml \
    --checkpoint_path ./checkpoints_aia_only/epoch_14.pth \
    --nc_dir $NC_DIR --hour_filter 0 \
    --output_csv ./test_results/aia_only_24h.csv 2>&1

echo "=========================================="
echo "8. HMI only 24h - epoch_14"
echo "=========================================="
python test.py \
    --config_path ./config_hmi_only.yaml \
    --checkpoint_path ./checkpoints_hmi_only/epoch_14.pth \
    --nc_dir $NC_DIR --hour_filter 0 \
    --output_csv ./test_results/hmi_only_24h.csv 2>&1

echo "=========================================="
echo "9. MobileNet scratch 24h - epoch_13"
echo "=========================================="
python test.py \
    --config_path ./config_mobilenet_scratch.yaml \
    --checkpoint_path ./checkpoints_mobilenet_scratch/epoch_13.pth \
    --nc_dir $NC_DIR --hour_filter 0 \
    --output_csv ./test_results/mobilenet_scratch_24h.csv 2>&1

echo "=========================================="
echo "PART 3 COMPLETE"
echo "=========================================="
