import argparse
import sys
import os

import numpy as np
import torch
import torch.distributed as dist
import wandb

# Now try imports
from dataset import ArDSDataset
from torch.amp import GradScaler, autocast
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models
from surya.utils import distributed
from surya.utils.log import log
import yaml
from typing import Union
import torch.nn as nn

from surya.utils.data import build_scalers
from surya.utils.distributed import (
    StatefulDistributedSampler,
    init_ddp,
    print0,
    save_model_singular,
    set_global_seed,
)

from peft import LoraConfig, get_peft_model
from models import HelioSpectformer2D, UNet, ChannelAdapter, LightweightSegModel


class DiceLoss(nn.Module):
    def __init__(self, smooth: Union[str, float] = 1e-6):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, preds, target):
        # Cast to float32 to avoid bfloat16 precision loss when summing millions of pixels
        preds = torch.sigmoid(preds.float()).view(-1)
        target = target.float().view(-1)

        intersection = (preds * target).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class IoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, preds, target):
        outputs = torch.sigmoid(preds.float())
        target = target.float()
        intersection = (outputs * target).sum(dim=(1, 2, 3))
        union = outputs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        return 1.0 - iou.mean()


def custom_collate_fn(batch):
    data_batch, metadata_batch = zip(*batch)

    try:
        collated_data = torch.utils.data.default_collate(data_batch)
    except TypeError:
        collated_data = data_batch

    if isinstance(metadata_batch[0], dict):
        collated_metadata = {}
        for key in metadata_batch[0].keys():
            values = [d[key] for d in metadata_batch]
            try:
                collated_metadata[key] = torch.utils.data.default_collate(values)
            except TypeError:
                collated_metadata[key] = values
    else:
        try:
            collated_metadata = torch.utils.data.default_collate(metadata_batch)
        except TypeError:
            collated_metadata = metadata_batch

    return collated_data, collated_metadata


def compute_segmentation_metrics(outputs, target, threshold=0.5):
    """
    Compute segmentation metrics: IoU, Dice, Precision, Recall.
    """
    preds = (torch.sigmoid(outputs) > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (preds * target_bin).sum()
    union = preds.sum() + target_bin.sum() - intersection
    pred_sum = preds.sum()
    target_sum = target_bin.sum()

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
    }


def evaluate_model(dataloader, epoch, model, device, run, criterion, total_steps=0):
    model.eval()

    abs_err_sum = torch.tensor(0.0, device=device)
    sq_err_sum = torch.tensor(0.0, device=device)
    targ_sum = torch.tensor(0.0, device=device)
    targ_sq_sum = torch.tensor(0.0, device=device)
    total_n = torch.tensor(0.0, device=device)
    running_loss, num_batches = 0.0, 0

    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0

    with torch.no_grad():
        for i, (batch, metadata) in enumerate(dataloader):
            curr_batch = {k: v.to(device) for k, v in batch.items()}
            if config["iters_per_epoch_valid"] == i:
                break

            with autocast(device_type="cuda", dtype=config["dtype"]):
                outputs = model(curr_batch)
                target = curr_batch["forecast"].unsqueeze(1)
                loss = criterion(outputs, target)

            reduced_loss = loss.detach()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= dist.get_world_size()

            running_loss += loss.item()
            num_batches += 1

            seg_metrics = compute_segmentation_metrics(outputs, target)
            total_iou += seg_metrics["iou"]
            total_dice += seg_metrics["dice"]
            total_precision += seg_metrics["precision"]
            total_recall += seg_metrics["recall"]

            if i % config["wandb_log_train_after"] == 0 and distributed.is_main_process():
                print0(f"Epoch: {epoch}, batch: {i}, loss: {reduced_loss.item()}")
                log(run, {"val_loss": reduced_loss.item()}, step=total_steps)

            diff = outputs - target
            abs_err_sum += torch.abs(diff).sum()
            sq_err_sum += (diff**2).sum()
            targ_sum += target.sum()
            targ_sq_sum += (target**2).sum()
            total_n += torch.tensor(target.numel(), device=device)

    for t in [abs_err_sum, sq_err_sum, targ_sum, targ_sq_sum, total_n]:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    mae = abs_err_sum.item() / total_n.item()
    mse = sq_err_sum.item() / total_n.item()
    rmse = mse**0.5

    var_y = (targ_sq_sum.item() - (targ_sum.item() ** 2) / total_n.item()) / total_n.item()
    r2 = float("nan") if var_y == 0 else 1.0 - (mse / var_y)

    avg_loss = running_loss / max(num_batches, 1)
    avg_iou = total_iou / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)
    avg_precision = total_precision / max(num_batches, 1)
    avg_recall = total_recall / max(num_batches, 1)

    if distributed.is_main_process():
        print0(
            f"Validation - Loss: {avg_loss:.4f}  "
            f"IoU: {avg_iou:.4f}  Dice: {avg_dice:.4f}  "
            f"Precision: {avg_precision:.4f}  Recall: {avg_recall:.4f}  "
            f"MAE: {mae:.4f}  RMSE: {rmse:.4f}"
        )
        log(
            run,
            {
                "valid/loss": avg_loss,
                "valid/iou": avg_iou,
                "valid/dice": avg_dice,
                "valid/precision": avg_precision,
                "valid/recall": avg_recall,
                "valid/mae": mae,
                "valid/rmse": rmse,
                "valid/r2": r2,
                "valid/total": int(total_n.item()),
            },
            step=total_steps,
        )

    return avg_loss, avg_iou, avg_dice


def wrap_all_checkpoints(model):
    for name, module in model.named_children():
        if (
            isinstance(module, torch.nn.Sequential)
            or isinstance(module, torch.nn.Linear)
            or isinstance(module, torch.nn.Conv2d)
        ):
            setattr(
                model,
                name,
                checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            )


def get_model(config, wandb_logger) -> torch.nn.Module:

    if torch.distributed.is_initialized() and distributed.is_main_process():
        print0("Creating the model.")

    if config["model"]["model_type"] == "spectformer_lora":
        print0("Initializing spectformer with LoRA.")
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
        print0("Initializing UNet.")
        model = UNet(
            in_chans=config["model"]["in_channels"],
            embed_dim=config["model"]["unet_embed_dim"],
            out_chans=1,
            n_blocks=config["model"]["unet_blocks"],
        )
    elif config["model"]["model_type"] == "mobilenet_deeplabv3":
        print0("Initializing MobileNetV3 + DeepLabV3.")
        model = LightweightSegModel(
            in_chans=config["model"]["in_channels"],
            out_chans=1,
            pretrained=config["model"].get("mobilenet_pretrained", True),
        )
    else:
        raise ValueError(f"Unknown model type {config['model']['model_type']}.")

    if torch.cuda.is_available():
        print0("GPU is available")
        device = torch.cuda.current_device()

    pretrained_path = config["pretrained_path"]

    if config["model"]["model_type"] == "spectformer":
        if (pretrained_path is not None) and os.path.exists(pretrained_path):
            print0(f"Loading pretrained model from {pretrained_path}.")
            model_state = model.state_dict()
            checkpoint_state = torch.load(pretrained_path, weights_only=True, map_location="cpu")

            filtered_checkpoint_state = {
                k: v
                for k, v in checkpoint_state.items()
                if k in model_state and v.shape == model_state[k].shape
            }

            model_state.update(filtered_checkpoint_state)
            model.load_state_dict(model_state, strict=True)

        else:
            raise ValueError(f"No checkpoint or pretrained model found at {pretrained_path}.")

    if torch.distributed.is_initialized() and distributed.is_main_process():
        active = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        total = sum(p.numel() for p in model.parameters()) / 1e6
        print0(f"MODEL: {active:.2f} M ACTIVE / {total:.2f} M TOTAL PARAMETERS.")

    return model


def apply_peft_lora(
    model: torch.nn.Module,
    config,
) -> torch.nn.Module:

    if not "lora_config" in config["model"]:
        print0("No LoRA configuration found. Using default LoRA settings.")
        lora_config = {
            "r": 32,
            "lora_alpha": 64,
            "target_modules": [
                "q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2",
            ],
            "lora_dropout": 0.1,
            "bias": "none",
        }
    else:
        lora_config = config["model"]["lora_config"]

    print0(f"Applying PEFT LoRA with configuration: {lora_config}")

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

    if distributed.is_main_process():
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print0(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.2f}%"
        )
    return model


def freeze_backbone(model, config):
    """Freeze all parameters except the head (unembed) for linear probing."""
    print0("Freezing backbone for linear probing - only training head (unembed)...")
    for name, param in model.named_parameters():
        if "unembed" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if distributed.is_main_process():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_param = sum(p.numel() for p in model.parameters())
        print0(
            f"After freeze - trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}%"
        )
    return model


def broadcast_dict(obj_dict, src=0):
    rank = torch.distributed.get_rank()
    if rank != src:
        obj_dict = None
    obj_list = [obj_dict]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def get_dataloaders(config, scalers):

    # Dataset always reads all 13 channels; ChannelAdapter selects channels at model level
    channels = config["data"]["channels"]

    train_dataset = ArDSDataset(
        zarr_path=config["data"]["train_zarr_path"],
        scalers=scalers,
        channels=channels,
        phase="train",
        pooling=config["data"].get("pooling", 1),
        ar_mask_base_dir=config["data"]["ar_mask_base_dir"],
        ds_ar_index_paths=config["data"]["ar_index_train"],
        daily_only=config["data"].get("daily_only", False),
        hour_filter=config["data"].get("hour_filter", None),
        year_start=config["data"].get("train_year_start", None),
        year_end=config["data"].get("train_year_end", None),
    )
    valid_dataset = ArDSDataset(
        zarr_path=config["data"]["valid_zarr_path"],
        scalers=scalers,
        channels=channels,
        phase="valid",
        pooling=config["data"].get("pooling", 1),
        ar_mask_base_dir=config["data"]["ar_mask_base_dir"],
        ds_ar_index_paths=config["data"]["ar_index_valid"],
        daily_only=config["data"].get("daily_only", False),
        hour_filter=config["data"].get("hour_filter", None),
    )

    print0(f"Train dataset size: {len(train_dataset)}")
    print0(f"Valid dataset size: {len(valid_dataset)}")

    dl_kwargs = dict(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_data_workers"],
        prefetch_factor=config["data"]["prefetch_factor"],
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=StatefulDistributedSampler(train_dataset, drop_last=True),
        **dl_kwargs,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        sampler=StatefulDistributedSampler(valid_dataset, drop_last=True),
        **dl_kwargs,
    )

    return train_loader, valid_loader


def main(config, use_gpu: bool, use_wandb: bool, profile: bool):

    run = None
    local_rank, rank = init_ddp(use_gpu)
    print0(f"RANK: {rank}; LOCAL_RANK: {local_rank}.")
    scalers = build_scalers(info=config["data"]["scalers"])
    os.makedirs(config["path_experiment"], exist_ok=True)

    if use_wandb and distributed.is_main_process():
        job_id = os.getenv("SLURM_JOB_ID", "unknown")
        print0(f"Job ID: {job_id}")
        print0(f"local_rank: {local_rank}, rank: {rank}: WANDB")

        run = wandb.init(
            project="ar-segmentation",
            entity="gsu-dmlab",
            name=f'[JOB: {job_id}] AR {config["job_id"]}',
            config=config,
            mode="online",
        )
        wandb.save(args.config_path)

    torch.distributed.barrier()

    train_loader, valid_loader = get_dataloaders(config, scalers)
    model = get_model(config, run)

    if config["model"]["use_lora"]:
        model = apply_peft_lora(model, config)

    if config["adapter"]["use_channel_adapter"]:
        adapter_channels = config["adapter"]["channels"]
        all_channels = config["data"]["channels"]
        channel_indices = [all_channels.index(ch) for ch in adapter_channels]
        num_data_chans = len(adapter_channels)
        print0("Using Adapters for", config["model"]["in_channels"], "-->", num_data_chans, "channels")
        print0("Channel indices:", channel_indices)
        model = ChannelAdapter(model,
            num_data_chans=num_data_chans,
            time_dim=config["model"]["time_embedding"]["time_dim"],
            channel_indices=channel_indices,
        )

    # Freeze backbone for linear probing if specified
    if config.get("freeze_backbone", False):
        model = freeze_backbone(model, config)

    model.to(rank)

    if len(config["model"]["checkpoint_layers"]) > 0:
        print0("Using checkpointing.")
        wrap_all_checkpoints(model)

    total_params = sum(p.numel() for p in model.parameters())
    print0(f"Total number of parameters: {total_params:,}")

    # Use find_unused_parameters=True for linear probing to avoid DDP errors
    find_unused = config.get("freeze_backbone", False)
    model = DistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()],
        find_unused_parameters=find_unused,
    )

    # Resume from checkpoint if specified
    resume_path = config.get("resume_checkpoint", None)
    start_epoch = config.get("start_epoch", 0)
    if resume_path and os.path.exists(resume_path):
        print0(f"Resuming from checkpoint: {resume_path}")
        checkpoint_state = torch.load(resume_path, map_location="cpu")
        model.module.load_state_dict(checkpoint_state)
        print0(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
    elif resume_path:
        print0(f"WARNING: resume_checkpoint specified but not found: {resume_path}")
        print0("Starting training from scratch.")
        start_epoch = 0

    # Loss selection
    loss_type = config["model"].get("select", "bce")
    if loss_type == "bce":
        criterion = torch.nn.BCEWithLogitsLoss()
        print0("Using BCE loss")
    elif loss_type == "dice":
        criterion = DiceLoss(smooth=float(config["model"]["dice"]["smooth"]))
        print0("Using Dice loss")
    elif loss_type == "bce_dice":
        bce = torch.nn.BCEWithLogitsLoss()
        dice = DiceLoss(smooth=float(config["model"]["dice"]["smooth"]))
        criterion = lambda pred, target: 0.5 * bce(pred, target) + 0.5 * dice(pred, target)
        print0("Using BCE + Dice combo loss")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Only optimize parameters that require grad (important for linear probing)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["optimizer"]["learning_rate"]
    )
    device = local_rank

    # Cosine annealing learning rate scheduler
    total_epochs = config["optimizer"]["max_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - start_epoch,
        eta_min=config["optimizer"]["min_lr"],
    )

    scaler = GradScaler()
    total_steps = start_epoch * min(config.get("iters_per_epoch_train", 99999), 99999)
    print0(f"Starting training for epochs {start_epoch} to {total_epochs - 1}.")
    print0(f"Learning rate: {config['optimizer']['learning_rate']} -> {config['optimizer']['min_lr']} (cosine annealing)")

    for epoch in range(start_epoch, total_epochs):
        print0(f"Epoch {epoch} of {total_epochs}")
        model.train()
        running_loss = torch.tensor(0.0, device=device)
        running_batch = torch.tensor(0, device=device)

        for i, (batch, metadata) in enumerate(train_loader):
            total_steps += 1
            if config["iters_per_epoch_train"] == i:
                break

            curr_batch = {k: v.to(local_rank) for k, v in batch.items()}

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=config["dtype"]):
                outputs = model(curr_batch)
                target = curr_batch["forecast"].unsqueeze(1)
                loss = criterion(outputs, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            reduced_loss = loss.detach()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= dist.get_world_size()

            running_loss += reduced_loss
            running_batch += 1

            if i % config["wandb_log_train_after"] == 0 and distributed.is_main_process():
                print0(f"Epoch: {epoch}, batch: {i}, loss: {reduced_loss.item()}")
                log(run, {"train_loss": reduced_loss.item()}, step=total_steps)

            if (i + 1) % config["save_wt_after_iter"] == 0:
                print0(f"Reached save_wt_after_iter ({config['save_wt_after_iter']}).")
                fp = os.path.join(config["path_experiment"], "checkpoint.pth")
                distributed.save_model_singular(model, fp, parallelism=config["parallelism"])

        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_batch, op=dist.ReduceOp.SUM)

        if distributed.is_main_process():
            log(run, {"epoch_loss": running_loss.item() / running_batch.item()}, step=total_steps)

        fp = os.path.join(config["path_experiment"], f"epoch_{epoch}.pth")
        save_model_singular(model, fp, parallelism=config["parallelism"])
        print0(f"Epoch {epoch}: Model saved at {fp}")

        evaluate_model(valid_loader, epoch, model, rank, run, criterion, total_steps)

        # Step the learning rate scheduler
        scheduler.step()
        if distributed.is_main_process():
            current_lr = scheduler.get_last_lr()[0]
            print0(f"Epoch {epoch}: Learning rate = {current_lr:.8f}")
            log(run, {"learning_rate": current_lr}, step=total_steps)

    if run is not None:
        run.finish()


if __name__ == "__main__":

    set_global_seed(0)

    parser = argparse.ArgumentParser("AR Segmentation Fine-tuning")
    parser.add_argument(
        "--config_path",
        default="./config.yaml",
        type=str,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument("--gpu", default=True, action="store_true", help="Run on GPU CUDA.")
    parser.add_argument("--wandb", default=False, action="store_true", help="Log into WanDB.")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, "r"))
    config["data"]["scalers"] = yaml.safe_load(open(config["data"]["scalers_path"], "r"))

    if config["dtype"] == "float16":
        config["dtype"] = torch.float16
    elif config["dtype"] == "bfloat16":
        config["dtype"] = torch.bfloat16
    elif config["dtype"] == "float32":
        config["dtype"] = torch.float32
    else:
        raise NotImplementedError("Please choose from [float16,bfloat16,float32]")

    if not args.gpu:
        raise ValueError(
            "Training scripts are not configured for CPU use. Please set the `--gpu` flag."
        )

    main(config=config, use_gpu=args.gpu, use_wandb=args.wandb, profile=args.profile)
    torch.distributed.destroy_process_group()