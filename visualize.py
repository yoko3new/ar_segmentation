import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

# Import from the same directory
from models import HelioSpectformer2D, UNet, LightweightSegModel
from peft import LoraConfig, get_peft_model
from dataset import ArDSDataset

# Import from surya
from surya.utils.data import build_scalers
from surya.utils.distributed import set_global_seed

# Reuse functions from infer.py
from infer import (
    custom_collate_fn,
    plot_sun_sdo_cmap2,
    format_metadata,
)
from finetune import custom_collate_fn


def get_model(config) -> torch.nn.Module:
    """Initialize and return the model based on the configuration."""
    print("Creating the model.")

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
        )
    else:
        raise ValueError(f"Unknown model type {config['model']['model_type']}.")

    return model


def apply_peft_lora(model, config):
    """Apply PEFT LoRA to the model."""
    lora_config = config["model"]["lora_config"]
    print(f"Applying PEFT LoRA with configuration: {lora_config}")

    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        target_modules=lora_config.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        ),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        bias=lora_config.get("bias", "none"),
    )

    model = get_peft_model(model, peft_config)
    return model


def load_model(config, checkpoint_path, device):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Initialize model
    model = get_model(config)

    # Apply LoRA if needed
    if config["model"].get("use_lora", False):
        model = apply_peft_lora(model, config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        model_state = checkpoint

    # Remove 'module.' prefix if present (from DistributedDataParallel)
    if any(key.startswith('module.') for key in model_state.keys()):
        model_state = {key.replace('module.', ''): value for key, value in model_state.items()}

    # Load state dict
    model.load_state_dict(model_state, strict=True)
    print("Model loaded successfully with strict=True")

    model.to(device)
    model.eval()

    return model


def get_dataloader(config, scalers, num_samples=3):
    """Create dataloader for visualization using ArDSDataset (zarr-based)."""

    dataset = ArDSDataset(
        zarr_path=config["data"]["valid_zarr_path"],
        scalers=scalers,
        channels=config["data"]["channels"],
        phase="valid",
        pooling=config["data"].get("pooling", 1),
        ar_mask_base_dir=config["data"]["ar_mask_base_dir"],
        ds_ar_index_paths=config["data"]["ar_index_valid"],
        daily_only=config["data"].get("daily_only", False),
    )

    print(f"Total dataset size: {len(dataset)}")
    assert len(dataset) > 0, "No data found"

    # Pick num_samples random samples (with seed for reproducibility)
    g = torch.Generator().manual_seed(42)
    random_ids = torch.randperm(len(dataset), generator=g)[:num_samples].tolist()
    print(f"Selected sample indices: {random_ids}")

    dataloader = DataLoader(
        dataset=Subset(dataset, random_ids),
        batch_size=1,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    return dataloader


def visualize_samples(
    model,
    dataloader,
    device,
    scalers,
    channels,
    dtype,
    output_dir,
    time_dim=1,
):
    """Run inference on samples and save visualizations including ground truth."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, (batch, metadata) in enumerate(dataloader):
            print(f"Processing sample {i}...")
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                pred = model(batch)
                if pred.ndim == 5:
                    pred = pred[:, 0]

                # Apply sigmoid to get probabilities
                pred = torch.sigmoid(pred)

                # Build visualization dict: input + prediction + ground truth
                def assort_all_channels(tensor):
                    """Expand single-channel mask to 13 channels for visualization."""
                    return torch.cat(
                        [
                            (
                                tensor[channels.index(channel), :, :][None, :, :]
                                if channel in channels
                                else torch.empty_like(tensor[0, :, :][None, :, :])
                            )
                            for channel in channels
                        ],
                        dim=0,
                    )

                save_dict = {}

                # Input images (13 channels)
                for input_idx in range(time_dim):
                    save_dict[f"input_{input_idx}"] = assort_all_channels(
                        batch["ts"][0, :, input_idx, :, :].squeeze().cpu()
                    )

                # Prediction (expand to 13 channels for plotting)
                pred_expanded = pred.expand(1, 13, -1, -1)
                save_dict["prediction"] = assort_all_channels(
                    pred_expanded[0, ...].squeeze().cpu()
                )

                # Ground truth (expand to 13 channels)
                target = batch["forecast"].unsqueeze(1)  # (1, 1, H, W)
                target_expanded = target.expand(1, 13, -1, -1)
                save_dict["ground_truth"] = assort_all_channels(
                    target_expanded[0, ...].squeeze().cpu()
                )

                # Format metadata for title
                metadata_str = format_metadata(metadata)

                # Save visualization
                save_path = os.path.join(output_dir, f"sample_{i}.png")
                plot_sun_sdo_cmap2(
                    list(save_dict.values()),
                    scaler=scalers,
                    save_path=save_path,
                    dpi=100,
                    scaler_keys=channels[:-5],  # AIA channels only for top row colormaps
                    title_txt=f"AR Segmentation - {metadata_str}",
                )
                print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser("AR Segmentation Visualization")
    parser.add_argument(
        "--config_path",
        default="./config_visualize.yaml",
        type=str,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        default="./visualization_results",
        type=str,
        help="Directory to save visualization results.",
    )
    parser.add_argument(
        "--num_samples",
        default=3,
        type=int,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to run inference on (cuda or cpu).",
    )
    args = parser.parse_args()

    set_global_seed(42)

    # Load config
    config = yaml.safe_load(open(args.config_path, "r"))
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))

    # Set dtype
    if config["dtype"] == "float16":
        config["dtype"] = torch.float16
    elif config["dtype"] == "bfloat16":
        config["dtype"] = torch.bfloat16
    elif config["dtype"] == "float32":
        config["dtype"] = torch.float32
    else:
        raise NotImplementedError("Please choose from [float16,bfloat16,float32]")

    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Build scalers
    scalers = build_scalers(info=config["data"]["scalers"])

    # Load model
    model = load_model(config, args.checkpoint_path, device)

    # Get dataloader
    dataloader = get_dataloader(config, scalers, num_samples=args.num_samples)

    # Run visualization
    visualize_samples(
        model=model,
        dataloader=dataloader,
        device=device,
        scalers=scalers,
        channels=config["data"]["channels"],
        dtype=config["dtype"],
        output_dir=args.output_dir,
        time_dim=config["model"]["time_embedding"]["time_dim"],
    )

    print(f"\nVisualization complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()