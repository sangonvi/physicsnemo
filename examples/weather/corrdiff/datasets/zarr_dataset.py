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
        indices=None,
        normalization_path=None,
        apply_input_normalization=True,
        apply_target_normalization=True,
    ):

        # ============================================================
        # OPEN ZARR
        # ============================================================

        self.root = zarr.open(
            path,
            mode="r",
        )

        self.inputs = self.root["input"]
        self.targets = self.root["target"]
        self.masks = self.root["mask"]

        # ============================================================
        # INDICES
        # ============================================================

        if indices is None:
            self.indices = np.arange(
                self.inputs.shape[0]
            )

        else:
            self.indices = np.load(indices)

        # ============================================================
        # NORMALIZATION FLAGS
        # ============================================================

        self.apply_input_normalization = (
            apply_input_normalization
        )

        self.apply_target_normalization = (
            apply_target_normalization
        )

        # ============================================================
        # LOAD NORMALIZATION
        # ============================================================

        self.input_mean = None
        self.input_std = None

        self.target_mean = None
        self.target_std = None

        if normalization_path is not None:

            norm = np.load(normalization_path)

            # --------------------------------------------------------
            # INPUT NORMALIZATION
            # --------------------------------------------------------

            self.input_mean = norm[
                "input_mean"
            ][:, None, None]

            self.input_std = norm[
                "input_std"
            ][:, None, None]

            # --------------------------------------------------------
            # TARGET NORMALIZATION
            # --------------------------------------------------------

            self.target_mean = norm[
                "target_mean"
            ][:, None, None]

            self.target_std = norm[
                "target_std"
            ][:, None, None]

    # ============================================================
    # LENGTH
    # ============================================================

    def __len__(self):

        return len(self.indices)

    # ============================================================
    # GET ITEM
    # ============================================================

    def __getitem__(self, idx):

        idx = self.indices[idx]

        # ------------------------------------------------------------
        # LOAD INPUT
        # ------------------------------------------------------------

        x = np.asarray(
            self.inputs[idx],
            dtype=np.float32,
        )

        # ------------------------------------------------------------
        # LOAD TARGET
        # ------------------------------------------------------------

        y = np.asarray(
            self.targets[idx],
            dtype=np.float32,
        )

        # ------------------------------------------------------------
        # LOAD MASK
        # ------------------------------------------------------------

        mask = np.asarray(
            self.masks[idx],
            dtype=np.float32,
        )

        # ============================================================
        # INPUT NORMALIZATION
        # ============================================================

        if (
            self.apply_input_normalization
            and self.input_mean is not None
        ):

            x = (
                x - self.input_mean
            ) / (
                self.input_std + 1e-6
            )

        # ============================================================
        # TARGET NORMALIZATION
        # ============================================================

        if (
            self.apply_target_normalization
            and self.target_mean is not None
        ):

            # --------------------------------------------------------
            # LOG1P TRANSFORM
            # --------------------------------------------------------

            y = np.log1p(y)

            # --------------------------------------------------------
            # Z-SCORE NORMALIZATION
            # --------------------------------------------------------

            y = (
                y - self.target_mean
            ) / (
                self.target_std + 1e-6
            )

        # ============================================================
        # APPLY MASK
        # ============================================================

        y = y * mask

        # ============================================================
        # RETURN
        # ============================================================

        return (
            torch.from_numpy(y),
            torch.from_numpy(x),
        )

    # ============================================================
    # CORRDIFF API
    # ============================================================

    def input_channels(self):

        return list(
            range(
                self.inputs.shape[1]
            )
        )

    def output_channels(self):

        return [0]

    def image_shape(self):

        return (
            self.inputs.shape[2],
            self.inputs.shape[3],
        )

    # ============================================================
    # OPTIONAL HELPERS
    # ============================================================

    def denormalize_target(
        self,
        y,
    ):

        """
        Convert normalized radar prediction
        back to physical reflectivity values.
        """

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        y = (
            y * self.target_std
        ) + self.target_mean

        y = np.expm1(y)

        return y

    def normalize_input(
        self,
        x,
    ):

        """
        Normalize ERA5 input manually.
        """

        return (
            x - self.input_mean
        ) / (
            self.input_std + 1e-6
        )

    def normalize_target(
        self,
        y,
    ):

        """
        Normalize radar target manually.
        """

        y = np.log1p(y)

        y = (
            y - self.target_mean
        ) / (
            self.target_std + 1e-6
        )

        return y