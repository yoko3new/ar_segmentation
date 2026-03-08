import os
import time
from datetime import datetime
from functools import cache
from logging import Logger

import h5py
import numpy as np
import pandas as pd
import skimage.measure
import torch
import zarr
from torch.utils.data import Dataset
from surya.utils.distributed import get_rank
from surya.utils.log import create_logger


def transform(
    data: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    sl_scale_factors: np.ndarray,
    epsilons: np.ndarray,
) -> np.ndarray:
    """
    Implements signum log transform.
    Args:
        data: Numpy array of shape C, H, W
        means, stds, sl_scale_factors, epsilons: Numpy arrays of shape C.
    Returns:
        Numpy array of shape C, H, W.
    """
    means = means.reshape(*means.shape, 1, 1)
    stds = stds.reshape(*stds.shape, 1, 1)
    sl_scale_factors = sl_scale_factors.reshape(*sl_scale_factors.shape, 1, 1)
    epsilons = epsilons.reshape(*epsilons.shape, 1, 1)

    data = data * sl_scale_factors
    data = np.sign(data) * np.log1p(np.abs(data))
    data = (data - means) / (stds + epsilons)

    return data


class ArDSDataset(Dataset):
    """
    AR Segmentation dataset that reads input images from zarr
    and binary masks from h5 files.
    """

    def __init__(
        self,
        zarr_path: str,
        scalers=None,
        channels: list[str] | None = None,
        phase: str = "train",
        pooling: int | None = None,
        ar_mask_base_dir: str = "./assets/surya-bench-ar-segmentation",
        ds_ar_index_paths: list = None,
    ):
        self.scalers = scalers
        self.phase = phase
        self.channels = channels
        self.pooling = pooling if pooling is not None else 1
        self.ar_mask_base_dir = ar_mask_base_dir

        # Default 13 channels
        if self.channels is None:
            self.channels = [
                'aia94', 'aia131', 'aia171', 'aia193', 'aia211',
                'aia304', 'aia335', 'aia1600',
                'hmi_m', 'hmi_bx', 'hmi_by', 'hmi_bz', 'hmi_v',
            ]
        self.in_channels = len(self.channels)

        # Open zarr
        t0 = time.perf_counter()
        self.data_zarr = zarr.open(zarr_path, mode="r")
        t1 = time.perf_counter()
        print(f"Zarr opened in {t1 - t0:.2f}s: {zarr_path}")

        # Build zarr timestamp index: timestep -> zarr array index
        ts = self.data_zarr["timestep"][:]
        self.zarr_index = pd.DataFrame({
            "timestep": pd.to_datetime(ts, unit="ns"),
            "zarr_idx": np.arange(ts.shape[0], dtype=int),
        })
        self.zarr_index.set_index("timestep", inplace=True)
        self.zarr_index.sort_index(inplace=True)

        # Load AR mask index (only rows where mask is present)
        all_data = [pd.read_csv(f) for f in ds_ar_index_paths]
        self.ar_index = pd.concat(all_data, ignore_index=True)
        self.ar_index = self.ar_index[self.ar_index["present"] == 1.0].copy()
        self.ar_index["timestamp"] = pd.to_datetime(self.ar_index["timestamp"])

        # Find intersection: zarr timestamps that have a matching mask
        zarr_timestamps = set(self.zarr_index.index)
        mask_timestamps = set(self.ar_index["timestamp"])
        matched = sorted(zarr_timestamps & mask_timestamps)

        # Build final valid samples
        self.ar_index.set_index("timestamp", inplace=True)
        self.valid_timestamps = matched
        self.adjusted_length = len(self.valid_timestamps)

        print(f"[{phase}] Zarr timestamps: {len(zarr_timestamps)}, "
              f"Mask files: {len(mask_timestamps)}, "
              f"Matched samples: {self.adjusted_length}")

        # Logger
        self.rank = get_rank()
        self.logger: Logger | None = None

    def create_logger(self):
        os.makedirs("logs/data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        pid = os.getpid()
        self.logger = create_logger(
            output_dir="logs/data",
            dist_rank=self.rank,
            name=f"{timestamp}_{self.rank:>03}_data_{self.phase}_{pid}",
        )

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> tuple:
        if self.logger is None:
            self.create_logger()

        timestep = self.valid_timestamps[idx]

        # --- Load input image from zarr ---
        zarr_idx = self.zarr_index.loc[timestep, "zarr_idx"]
        img_data = self.data_zarr["img"][zarr_idx]  # (13, 4096, 4096)
        img_data = self.transform_data(img_data)
        # Add time dimension: (C, H, W) -> (C, 1, H, W)
        img_data = np.stack([img_data], axis=1)

        # --- Load mask from h5 ---
        file_path = self.ar_index.loc[timestep, "file_path"]
        full_path = os.path.join(self.ar_mask_base_dir, file_path)

        try:
            with h5py.File(full_path, "r") as f:
                mask = torch.from_numpy(f["union_with_intersect"][...])
        except Exception as e:
            print(f"Error loading mask from {full_path}: {e}")
            raise e

        # --- Build output dictionary ---
        base_dictionary = {
            "ts": img_data,
            "time_delta_input": torch.tensor([0]),
            "forecast": mask / 255.0,
        }

        metadata = {
            "timestamps_input": np.datetime64(timestep),
            "timestamps_targets": np.datetime64(timestep),
        }

        return base_dictionary, metadata

    @cache
    def transformation_inputs(self):
        means = np.array([self.scalers[ch].mean for ch in self.channels])
        stds = np.array([self.scalers[ch].std for ch in self.channels])
        epsilons = np.array([self.scalers[ch].epsilon for ch in self.channels])
        sl_scale_factors = np.array(
            [self.scalers[ch].sl_scale_factor for ch in self.channels]
        )
        return means, stds, epsilons, sl_scale_factors

    def transform_data(self, data: np.ndarray) -> np.ndarray:
        """Apply signum log transform to input data."""
        assert data.ndim == 3

        if self.pooling > 1:
            data = skimage.measure.block_reduce(
                data, block_size=(1, self.pooling, self.pooling), func=np.mean
            )

        means, stds, epsilons, sl_scale_factors = self.transformation_inputs()
        return transform(data, means, stds, sl_scale_factors, epsilons)