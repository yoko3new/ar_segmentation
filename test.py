"""
Test evaluation script for AR segmentation.
Reads SDO data from NetCDF files and AR masks from h5 files.
Computes IoU, Dice, Precision, Recall on the test set (both sample-wise and global).

Usage:
    # 24h models (daily only)
    python test.py --config_path ./config.yaml \
                   --checkpoint_path ~/ar_seg_checkpoints/spectformer/epoch_31.pth \
                   --hour_filter 0 \
                   --output_csv ./test_results/lora_bce_24h.csv

    # 12h models
    python test.py --config_path ./config_12h_bce_dice.yaml \
                   --checkpoint_path ~/ar_seg_checkpoints/bce_dice/epoch_25_12h_resume.pth \
                   --hour_filter 0 12 \
                   --output_csv ./test_results/lora_bce_dice_12h.csv

    # Quick debug (10 samples)
    python test.py --config_path ./config.yaml \
                   --checkpoint_path ~/ar_seg_checkpoints/spectformer/epoch_31.pth \
                   --hour_filter 0 --max_samples 10
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import xarray as xr
import yaml
from tqdm import tqdm

from models import HelioSpectformer2D, UNet, LightweightSegModel, ChannelAdapter
from peft import LoraConfig, get_peft_model
from surya.utils.data import build_scalers
from surya.utils.distributed import set_global_seed


def get_model(config):
    """Initialize the model based on config."""
    if config["model"]["model_type"] == "spectformer_lora":
        print("Initializing spectformer with LoRA.")
        model = HelioSpectformer2D(
            img_size=config["model"]["img_size"],
            patch_size=config["model"]["patch_size"],
            in_chans=config["model"]["in_channels"],
            embed_dim=config["model"]["embed_dim"],
            time_embedding=config["model"]["time_embedding"],
            depth=config["model"]["depth"],
            n_spectral_blocks=config["model"]["spectral_blocks"],
            num_heads=config["model"]["num_heads"],
            mlp_ratio=config["model"]["mlp_ratio"],
            drop_rate=config["model"]["drop_rate"],
            dtype=config["dtype"],
            window_size=config["model"]["window_size"],
            dp_rank=config["model"]["dp_rank"],
            learned_flow=config["model"]["learned_flow"],
            use_latitude_in_learned_flow=config["use_latitude_in_learned_flow"],
            init_weights=config["model"]["init_weights"],
            checkpoint_layers=config["model"]["checkpoint_layers"],
            rpe=config["model"]["rpe"],
            finetune=config["model"]["finetune"],
            config=config,
        )
    elif config["model"]["model_type"] == "unet":
        print("Initializing UNet.")
        model = UNet(
            in_chans=config["model"]["in_channels"],
            embed_dim=config["model"]["unet_embed_dim"],
            out_chans=1,
            n_blocks=config["model"]["unet_blocks"],
        )
    elif config["model"]["model_type"] == "mobilenet_deeplabv3":
        print("Initializing MobileNetV3 + DeepLabV3.")
        model = LightweightSegModel(
            in_chans=config["model"]["in_channels"],
            out_chans=1,
            pretrained=config["model"].get("mobilenet_pretrained", True),
        )
    else:
        raise ValueError(f"Unknown model type {config['model']['model_type']}.")
    return model


def apply_peft_lora(model, config):
    """Apply LoRA to model."""
    lora_config = config["model"].get("lora_config", {
        "r": 32, "lora_alpha": 64,
        "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        "lora_dropout": 0.1, "bias": "none",
    })
    print(f"Applying PEFT LoRA: {lora_config}")
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get("target_modules",
            ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        bias=lora_config.get("bias", "none"),
    )
    model = get_peft_model(model, peft_config)
    return model


def load_model(config, checkpoint_path, device):
    """Load model with checkpoint."""
    model = get_model(config)

    if config["model"].get("use_lora", False):
        model = apply_peft_lora(model, config)

    # Add ChannelAdapter if needed
    if config["adapter"]["use_channel_adapter"]:
        adapter_channels = config["adapter"]["channels"]
        all_channels = config["data"]["channels"]
        channel_indices = [all_channels.index(ch) for ch in adapter_channels]
        num_data_chans = len(adapter_channels)
        print(f"Using ChannelAdapter: {config['model']['in_channels']} --> {num_data_chans} channels")
        print(f"Channel indices: {channel_indices}")
        model = ChannelAdapter(model,
            num_data_chans=num_data_chans,
            time_dim=config["model"]["time_embedding"]["time_dim"],
            channel_indices=channel_indices,
        )

    # Freeze backbone for linear probing configs
    if config.get("freeze_backbone", False):
        print("Freezing backbone (linear probing mode).")
        for name, param in model.named_parameters():
            if "unembed" not in name:
                param.requires_grad = False

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint

    # Remove 'module.' prefix if present (from DDP)
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    print("Model loaded successfully.")

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total:.2f}M")

    model.to(device)
    model.eval()
    return model


def transform_data(data, scalers, channels):
    """Apply signum-log normalization (same as dataset.py)."""
    means = np.array([scalers[ch].mean for ch in channels])
    stds = np.array([scalers[ch].std for ch in channels])
    epsilons = np.array([scalers[ch].epsilon for ch in channels])
    sl_scale_factors = np.array([scalers[ch].sl_scale_factor for ch in channels])

    data = np.sign(data) * np.log1p(np.abs(sl_scale_factors[:, None, None] * data))
    data = (data - means[:, None, None]) / (stds[:, None, None] + epsilons[:, None, None])
    return data


def load_nc_sample(nc_path, channels):
    """Load a single NC file and return (13, 4096, 4096) numpy array."""
    ds = xr.open_dataset(nc_path)
    data = np.stack([ds[ch].values for ch in channels], axis=0)
    ds.close()
    return data.astype(np.float32)


def load_mask(mask_path):
    """Load AR mask from h5 file, return normalized (4096, 4096) tensor."""
    with h5py.File(mask_path, "r") as f:
        mask = f["union_with_intersect"][...]
    return torch.from_numpy(mask).float() / 255.0


def compute_metrics(preds, target, threshold=0.5):
    """Compute IoU, Dice, Precision, Recall + raw counts for global metrics."""
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (preds_bin * target_bin).sum()
    pred_sum = preds_bin.sum()
    target_sum = target_bin.sum()
    union = pred_sum + target_sum - intersection

    eps = 1e-7
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    precision = (intersection + eps) / (pred_sum + eps)
    recall = (intersection + eps) / (target_sum + eps)

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "tp": intersection.item(),
        "fp": (pred_sum - intersection).item(),
        "fn": (target_sum - intersection).item(),
    }


def build_test_samples(test_csv, nc_dir, mask_dir, year_start=None, year_end=None, hour_filter=None):
    """
    Build list of (nc_path, mask_path, timestamp) tuples from test.csv.
    Only includes samples where both NC file and mask file exist.
    """
    df = pd.read_csv(test_csv)
    df = df[df["present"] == 1.0].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if year_start is not None:
        df = df[df["timestamp"].dt.year >= year_start]
    if year_end is not None:
        df = df[df["timestamp"].dt.year <= year_end]

    if hour_filter is not None:
        df = df[df["timestamp"].dt.hour.isin(hour_filter)]
        print(f"Filtered to hours {hour_filter}: {len(df)} samples")

    samples = []
    missing_nc = 0
    missing_mask = 0
    for _, row in df.iterrows():
        ts = row["timestamp"]
        nc_filename = ts.strftime("%Y%m%d_%H%M") + ".nc"
        nc_path = os.path.join(nc_dir, str(ts.year), f"{ts.month:02d}", nc_filename)

        mask_rel_path = row["file_path"]
        mask_path = os.path.join(mask_dir, mask_rel_path)

        if not os.path.exists(nc_path):
            missing_nc += 1
            continue
        if not os.path.exists(mask_path):
            missing_mask += 1
            continue

        samples.append((nc_path, mask_path, str(ts)))

    if missing_nc > 0:
        print(f"Warning: {missing_nc} NC files not found")
    if missing_mask > 0:
        print(f"Warning: {missing_mask} mask files not found")

    return samples


def main():
    parser = argparse.ArgumentParser("AR Segmentation Test Evaluation")
    parser.add_argument("--config_path", default="./config.yaml", type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--nc_dir", default="/anvil/scratch/x-kyang10/test_nc", type=str)
    parser.add_argument("--test_csv", default="./assets/surya-bench-ar-segmentation/test.csv", type=str)
    parser.add_argument("--mask_dir", default="./assets/surya-bench-ar-segmentation", type=str)
    parser.add_argument("--year_start", default=None, type=int)
    parser.add_argument("--year_end", default=None, type=int)
    parser.add_argument("--hour_filter", nargs='+', type=int, default=None,
                        help="Hours to include, e.g. --hour_filter 0 for daily, --hour_filter 0 12 for 12h")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max_samples", default=None, type=int, help="Limit number of samples (for debugging)")
    parser.add_argument("--output_csv", default=None, type=str, help="Save per-sample results to CSV")
    parser.add_argument("--save_preds_dir", default=None, type=str, help="Save raw sigmoid predictions as .npy files (float16)")
    args = parser.parse_args()

    set_global_seed(42)

    # Load config
    config = yaml.safe_load(open(args.config_path, "r"))
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))

    if config["dtype"] == "float16":
        config["dtype"] = torch.float16
    elif config["dtype"] == "bfloat16":
        config["dtype"] = torch.bfloat16
    elif config["dtype"] == "float32":
        config["dtype"] = torch.float32

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Build scalers
    scalers = build_scalers(info=config["data"]["scalers"])
    channels = config["data"]["channels"]

    # Load model
    model = load_model(config, args.checkpoint_path, device)

    # Build test samples
    print(f"\nBuilding test samples from {args.test_csv}...")
    print(f"NC dir: {args.nc_dir}")
    print(f"Mask dir: {args.mask_dir}")
    if args.hour_filter:
        print(f"Hour filter: {args.hour_filter}")

    samples = build_test_samples(
        args.test_csv, args.nc_dir, args.mask_dir,
        year_start=args.year_start, year_end=args.year_end,
        hour_filter=args.hour_filter,
    )
    print(f"Found {len(samples)} valid test samples.")

    if len(samples) == 0:
        print("ERROR: No valid test samples found. Check paths and filters.")
        return

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limiting to {args.max_samples} samples.")

    # Run evaluation
    all_metrics = []
    with torch.no_grad():
        for i, (nc_path, mask_path, timestamp) in enumerate(tqdm(samples, desc="Evaluating")):
            try:
                # Skip if raw prediction already exists
                if args.save_preds_dir:
                    ts_str = timestamp.replace(" ", "_").replace(":", "").replace("-", "")
                    pred_path = os.path.join(args.save_preds_dir, f"{ts_str}.npy")
                    if os.path.exists(pred_path):
                        # Load existing prediction to compute metrics
                        sigmoid_out = torch.from_numpy(np.load(pred_path).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                        mask = load_mask(mask_path).unsqueeze(0).unsqueeze(0).to(device)
                        preds_bin = (sigmoid_out > 0.5).float()
                        target_bin = (mask > 0.5).float()
                        intersection = (preds_bin * target_bin).sum()
                        pred_sum = preds_bin.sum()
                        target_sum = target_bin.sum()
                        union = pred_sum + target_sum - intersection
                        eps = 1e-7
                        metrics = {
                            "iou": ((intersection+eps)/(union+eps)).item(),
                            "dice": ((2*intersection+eps)/(pred_sum+target_sum+eps)).item(),
                            "precision": ((intersection+eps)/(pred_sum+eps)).item(),
                            "recall": ((intersection+eps)/(target_sum+eps)).item(),
                            "tp": intersection.item(),
                            "fp": (pred_sum-intersection).item(),
                            "fn": (target_sum-intersection).item(),
                            "timestamp": timestamp,
                        }
                        all_metrics.append(metrics)
                        continue

                # Load and transform input
                data = load_nc_sample(nc_path, channels)
                data = transform_data(data, scalers, channels)

                # Add time dimension: (C, H, W) -> (C, 1, H, W)
                data = data[:, np.newaxis, :, :]

                # Build batch dict
                batch = {
                    "ts": torch.from_numpy(data).unsqueeze(0).to(device),
                    "time_delta_input": torch.tensor([[0]]).to(device),
                }

                # Forward pass
                with torch.amp.autocast(device_type="cuda", dtype=config["dtype"]):
                    outputs = model(batch)

                # Load mask
                mask = load_mask(mask_path).unsqueeze(0).unsqueeze(0).to(device)

                # Compute metrics
                metrics = compute_metrics(outputs, mask)
                metrics["timestamp"] = timestamp
                all_metrics.append(metrics)

                # Save raw sigmoid predictions if requested
                if args.save_preds_dir:
                    os.makedirs(args.save_preds_dir, exist_ok=True)
                    sigmoid_out = torch.sigmoid(outputs).cpu().squeeze().half().numpy()
                    ts_str = timestamp.replace(" ", "_").replace(":", "").replace("-", "")
                    np.save(os.path.join(args.save_preds_dir, f"{ts_str}.npy"), sigmoid_out)

                # Free GPU memory
                del outputs, mask, batch
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error on sample {i} ({timestamp}): {e}")
                continue

            if (i + 1) % 200 == 0:
                avg_iou = np.mean([m["iou"] for m in all_metrics])
                avg_dice = np.mean([m["dice"] for m in all_metrics])
                print(f"  [{i+1}/{len(samples)}] Running avg — IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")

    # Sample-wise (macro) averages
    avg_iou = np.mean([m["iou"] for m in all_metrics])
    avg_dice = np.mean([m["dice"] for m in all_metrics])
    avg_precision = np.mean([m["precision"] for m in all_metrics])
    avg_recall = np.mean([m["recall"] for m in all_metrics])

    # Global (micro) averages
    total_tp = sum(m["tp"] for m in all_metrics)
    total_fp = sum(m["fp"] for m in all_metrics)
    total_fn = sum(m["fn"] for m in all_metrics)
    eps = 1e-7
    global_iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    global_dice = (2 * total_tp + eps) / (2 * total_tp + total_fp + total_fn + eps)
    global_precision = (total_tp + eps) / (total_tp + total_fp + eps)
    global_recall = (total_tp + eps) / (total_tp + total_fn + eps)

    print("\n" + "=" * 60)
    print(f"TEST SET EVALUATION RESULTS ({len(all_metrics)} samples)")
    print("=" * 60)
    print(f"  Sample-wise (macro) average:")
    print(f"    IoU:       {avg_iou:.4f}")
    print(f"    Dice:      {avg_dice:.4f}")
    print(f"    Precision: {avg_precision:.4f}")
    print(f"    Recall:    {avg_recall:.4f}")
    print(f"  Global (micro) average:")
    print(f"    IoU:       {global_iou:.4f}")
    print(f"    Dice:      {global_dice:.4f}")
    print(f"    Precision: {global_precision:.4f}")
    print(f"    Recall:    {global_recall:.4f}")
    print("=" * 60)

    # Save per-sample results if requested
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else ".", exist_ok=True)
        results_df = pd.DataFrame(all_metrics)
        results_df.to_csv(args.output_csv, index=False)
        print(f"Per-sample results saved to {args.output_csv}")


if __name__ == "__main__":
    main()