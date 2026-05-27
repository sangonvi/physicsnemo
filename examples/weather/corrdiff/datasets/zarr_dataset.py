"""
dataset.py

Final Zarr dataset loader for CorrDiff training.

This implementation supports:

    - Zarr datasets
    - Train/validation split indices
    - ERA5 normalization
    - Radar log1p normalization
    - CorrDiff regression training
    - CorrDiff diffusion training

Expected dataset structure:

datasets/
└── corrdiff/
    ├── train.zarr/
    │   ├── input
    │   ├── target
    │   └── mask
    │
    ├── normalization.npz
    ├── train_index.npy
    └── valid_index.npy

Normalization strategy:

INPUT (ERA5):
    x = (x - mean) / std

TARGET (Radar):
    y = log1p(y)
    y = (y - mean) / std

During inference:

    y = y * std + mean
    y = expm1(y)

Author:
    Vinicius / CorrDiff adaptation

"""

import zarr
import torch
import numpy as np

from torch.utils.data import Dataset


class ZarrCorrDiffDataset(Dataset):

    def __init__(
        self,
        path,
        normalization_path=None,
        train_indices=None,
        valid_indices=None,
        mode="train",
        indices=None,
        **kwargs,
    ):

        print("LEN DATASET:", len(self))
        self.root = zarr.open(path, mode="r")

        self.inputs = self.root["input"]
        self.targets = self.root["target"]

        # ==========================================
        # SPLITS
        # ==========================================

        if indices is not None:

            self.indices = np.load(indices)

        elif mode == "train" and train_indices is not None:

            self.indices = np.load(train_indices)

        elif mode in ["valid", "validation"] and valid_indices is not None:

            self.indices = np.load(valid_indices)

        else:

            self.indices = np.arange(self.inputs.shape[0])

        # ==========================================
        # NORMALIZATION
        # ==========================================

        self.mean = None
        self.std = None

        if normalization_path is not None:

            norm = np.load(normalization_path)

            # ==========================================
            # COMPATIBLE NORMALIZATION LOADING
            # ==========================================

            if "mean" in norm.files:
                mean = norm["mean"]
            elif "input_mean" in norm.files:
                mean = norm["input_mean"]
            elif "x_mean" in norm.files:
                mean = norm["x_mean"]
            else:
                raise ValueError(
                    f"No mean field found in normalization file: {norm.files}"
                )

            if "std" in norm.files:
                std = norm["std"]
            elif "input_std" in norm.files:
                std = norm["input_std"]
            elif "x_std" in norm.files:
                std = norm["x_std"]
            else:
                raise ValueError(
                    f"No std field found in normalization file: {norm.files}"
                )

            self.mean = mean[:, None, None]
            self.std = std[:, None, None]
            
    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):

        idx = self.indices[idx]
        print("IDX:", idx)
        x = np.asarray(
            self.inputs[idx],
            dtype=np.float32,
        )

        y = np.asarray(
            self.targets[idx],
            dtype=np.float32,
        )

        # ==========================================
        # NORMALIZATION
        # ==========================================

        if self.mean is not None:

            x = (x - self.mean) / (self.std + 1e-6)

        return (
            torch.from_numpy(y),
            torch.from_numpy(x),
        )

    # ==========================================
    # METADATA
    # ==========================================

    def input_channels(self):

        return list(range(self.inputs.shape[1]))

    def output_channels(self):

        return [0]

    def image_shape(self):

        return (
            self.inputs.shape[2],
            self.inputs.shape[3],
        )