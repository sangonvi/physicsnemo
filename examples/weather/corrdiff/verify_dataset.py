from physicsnemo import Module
import torch
from datasets.zarr_dataset import ZarrCorrDiffDataset

CHECKPOINT = (
    "checkpoints/regression/checkpoints_regression/"
    "CorrDiffRegressionUNet.0.100000.mdlus"
)

DATASET_PATH = "/home/vsantos/rionowcast/datasets/corrdiff/train.zarr"

NORMALIZATION = "/home/vsantos/rionowcast/datasets/corrdiff/normalization.npz"

VALID_INDICES = "/home/vsantos/rionowcast/datasets/corrdiff/valid_index.npy"


print("Loading model...")

print("Loading validation dataset...")

dataset = ZarrCorrDiffDataset(
    path=DATASET_PATH,
    normalization_path=NORMALIZATION,
    valid_indices=VALID_INDICES,
    mode="validation",
)

model = Module.from_checkpoint(
    "checkpoints/regression/checkpoints_regression/CorrDiffRegressionUNet.0.100000.mdlus"
)

model.cpu()
model.eval()


max_std = 0
max_idx = 0

for i in range(len(dataset)):

    target, _ = dataset[i]

    s = target.std().item()

    if s > max_std:
        max_std = s
        max_idx = i

print("max_idx =", max_idx)
print("max_std =", max_std)

target, era5 = dataset[max_idx]

print(target.min())
print(target.max())
print(target.std())