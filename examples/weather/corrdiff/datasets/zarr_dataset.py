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
    ):

        self.root = zarr.open(path, mode="r")

        self.inputs = self.root["input"]
        self.targets = self.root["target"]

        if indices is None:
            self.indices = np.arange(self.inputs.shape[0])
        else:
            self.indices = np.load(indices)

        self.mean = None
        self.std = None

        if normalization_path is not None:
            norm = np.load(normalization_path)

            self.mean = norm["mean"][:, None, None]
            self.std = norm["std"][:, None, None]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        idx = self.indices[idx]

        x = np.asarray(
            self.inputs[idx],
            dtype=np.float32,
        )

        y = np.asarray(
            self.targets[idx],
            dtype=np.float32,
        )

        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-6)

        return (
            torch.from_numpy(y),
            torch.from_numpy(x),
        )

    def input_channels(self):
        return list(range(self.inputs.shape[1]))

    def output_channels(self):
        return [0]

    def image_shape(self):
        return (
            self.inputs.shape[2],
            self.inputs.shape[3],
        )