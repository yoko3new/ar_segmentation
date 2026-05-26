# Active Region Segmentation with Surya Foundation Model
 
Fine-tuning and evaluation of the Surya solar foundation model for Active Region (AR) segmentation, with comparisons to UNet and MobileNet+DeepLabV3 baselines.
 
## Task
 
Binary semantic segmentation of Active Regions containing Polarity Inversion Lines (PILs) from full-disk SDO observations.
 
- **Input**: 13-channel solar images (8 AIA + 5 HMI), 4096×4096 resolution
- **Output**: Binary mask (4096×4096), AR with PIL = 1, background = 0

### Model Parameters
 
| Model | Overall Parameters | Trainable Parameters |
|-------|-------------------|---------------------|
| SpectFormer+LoRA | 362.09M | 4.43M (1.22%) |
| Linear Probing | 358.00M | 0.33M (0.09%) |
| UNet (scratch) | 37.02M | 37.02M (100%) |
| MobileNet | 11.02M | 11.02M (100%) |
 
## Setup
 
### Prerequisites
- Python 3.11+
- CUDA-capable GPU (H100 80GB recommended)
- ~60GB disk space for AR masks
### Installation
```bash
conda create -n surya python=3.11
conda activate surya
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install zarr wandb peft h5py scikit-image pyyaml pandas tqdm einops timm xarray netCDF4
```
 
### Data Download
```bash
# Download AR segmentation masks and index files
bash download_data.sh
 
# Extract masks
cd assets/surya-bench-ar-segmentation
mkdir -p data
tar -xzf data.tar.gz -C data/
 
# Download test NC files from S3
aws s3 sync s3://nasa-surya-bench/2020/ test_nc/2020/ --no-sign-request --exclude "*" --include "*_0000.nc"
# Repeat for 2021-2024
```
 
## Training
 
### Surya + LoRA (default)
```bash
export PYTHONPATH=/path/to/Surya:$PYTHONPATH
export MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
 
python -u finetune.py --gpu --wandb --config_path ./config.yaml
```
 
### Configuration Options
 
| Config | Description |
|--------|-------------|
| `config.yaml` | LoRA+BCE, 24h daily, 2011-2017 |
| `config_unet.yaml` | UNet from scratch |
| `config_mobilenet.yaml` | MobileNet+DeepLabV3 (ImageNet pretrained) |
| `config_mobilenet_scratch.yaml` | MobileNet+DeepLabV3 (random init) |
| `config_linear_probe_24h.yaml` | Frozen backbone + linear head |
| `config_aia_only.yaml` | LoRA with AIA channels only (8ch) |
| `config_hmi_only.yaml` | LoRA with HMI channels only (5ch) |
| `config_12h_bce_dice.yaml` | LoRA+BCE+Dice, 12h cadence |
 
### Key Training Features
- **Resume training**: Set `resume_checkpoint` and `start_epoch` in config
- **Cosine annealing**: LR decays from `learning_rate` to `min_lr`
- **Loss functions**: `select: bce`, `dice`, or `bce_dice`
- **Freeze backbone**: `freeze_backbone: true` for linear probing
- **Channel ablation**: `adapter.use_channel_adapter: true` with selected channels
- **Hour filtering**: `data.hour_filter: [0, 12]` for temporal cadence control
## Test Evaluation
 
```bash
# 24h models
python test.py \
    --config_path ./config.yaml \
    --checkpoint_path ./best_checkpoints/lora_bce_epoch_31.pth \
    --nc_dir /path/to/test_nc \
    --hour_filter 0 \
    --output_csv ./test_results/lora_bce.csv \
    --save_preds_dir ./test_preds/lora_bce
 
# Quick debug (10 samples)
python test.py --config_path ./config.yaml \
    --checkpoint_path ./best_checkpoints/lora_bce_epoch_31.pth \
    --hour_filter 0 --max_samples 10
```
 
### Test Output
- **CSV**: Per-sample IoU, Dice, Precision, Recall, TP, FP, FN
- **Raw predictions**: Sigmoid outputs saved as float16 `.npy` files
- **Summary**: Both sample-wise (macro) and global (micro) averages
### Threshold Analysis
Raw sigmoid predictions enable post-hoc threshold analysis:
```bash
# Generate P-R curve data across 19 thresholds
python generate_pr_curve.py  # outputs test_results/pr_curve_data.csv
```
 
## File Structure
 
```
ar_segmentation/
├── finetune.py              # Training script
├── test.py                  # Test evaluation (macro + micro metrics)
├── dataset.py               # Dataset class (zarr input + h5 masks)
├── models.py                # Model definitions (SpectFormer, UNet, MobileNet, ChannelAdapter)
├── infer.py                 # Inference visualization
├── visualize.py             # Visualization utilities
├── create_ar_csv.py         # Dataset index generation
├── config*.yaml             # Training configurations
├── run_*.sh                 # Slurm job scripts
├── test_results/            # Per-sample CSVs, P-R curve data, plots
│   ├── lora_bce.csv
│   ├── mobilenet_scratch.csv
│   ├── pr_curve_data.csv
│   └── pr_curve_and_iou.png
├── test_preds/              # Raw sigmoid predictions (.npy, float16)
│   ├── lora_bce/
│   ├── mobilenet_scratch/
│   └── ...
├── best_checkpoints/        # Best model weights
│   ├── lora_bce_epoch_31.pth
│   ├── mobilenet_imagenet_epoch_14.pth
│   ├── mobilenet_scratch_epoch_13.pth
│   └── hmi_only_epoch_14.pth
└── assets/
    ├── surya.366m.v1.pt     # Surya pretrained weights
    ├── scalers.yaml         # Channel normalization parameters
    └── surya-bench-ar-segmentation/
        ├── data/            # AR masks (h5 files)
        ├── train.csv
        ├── validation.csv
        └── test.csv
```