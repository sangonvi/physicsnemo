from physicsnemo import Module
import torch
from datasets.zarr_dataset import ZarrCorrDiffDataset
import numpy as np

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

all_mae = []
all_rmse = []
all_corr = []

for idx in range(len(dataset)):

    target, era5 = dataset[idx]

    with torch.no_grad():

        pred = model(
            x=torch.zeros_like(target).unsqueeze(0),
            img_lr=era5.unsqueeze(0)
        )

    target = target.numpy().squeeze()
    pred = pred.numpy().squeeze()

    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(np.mean((pred - target) ** 2))

    all_mae.append(mae)
    all_rmse.append(rmse)

    if target.std() > 1e-6 and pred.std() > 1e-6:
        corr = np.corrcoef(
            pred.flatten(),
            target.flatten()
        )[0,1]

        all_corr.append(corr)

print()
print("Mean MAE :", np.mean(all_mae))
print("Mean RMSE:", np.mean(all_rmse))
print("Mean CORR:", np.mean(all_corr))