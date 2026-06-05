import numpy as np
import torch

from physicsnemo import Module
from datasets.zarr_dataset import ZarrCorrDiffDataset

DEVICE = "cpu"

CHECKPOINT = (
    "checkpoints/regression/checkpoints_regression/"
    "CorrDiffRegressionUNet.0.100000.mdlus"
)

DATASET_PATH = "/home/vsantos/rionowcast/datasets/corrdiff/train.zarr"

NORMALIZATION = "/home/vsantos/rionowcast/datasets/corrdiff/normalization.npz"

VALID_INDICES = "/home/vsantos/rionowcast/datasets/corrdiff/valid_index.npy"

print("Loading model...")

model = Module.from_checkpoint(CHECKPOINT)

model.eval()
model.to(DEVICE)

print("Loading dataset...")

dataset = ZarrCorrDiffDataset(
    path=DATASET_PATH,
    normalization_path=NORMALIZATION,
    valid_indices=VALID_INDICES,
    mode="validation",
)

print("Validation samples:", len(dataset))

norm = np.load(NORMALIZATION)

target_mean = norm["target_mean"][0]
target_std = norm["target_std"][0]

all_mae = []
all_rmse = []
all_corr = []
all_skill = []

for idx in range(len(dataset)):

    target, era5 = dataset[idx]

    with torch.no_grad():

        pred = model(
            x=torch.zeros_like(target).unsqueeze(0).to(DEVICE),
            img_lr=era5.unsqueeze(0).to(DEVICE)
        )

    target = target.numpy().squeeze()
    pred = pred.cpu().numpy().squeeze()

    target = target * target_std + target_mean
    pred = pred * target_std + target_mean

    mae = np.mean(np.abs(pred - target))

    rmse = np.sqrt(
        np.mean((pred - target) ** 2)
    )

    all_mae.append(mae)
    all_rmse.append(rmse)

    if target.std() > 1e-6 and pred.std() > 1e-6:

        corr = np.corrcoef(
            pred.flatten(),
            target.flatten()
        )[0,1]

        all_corr.append(corr)

        baseline_rmse = np.sqrt(
            np.mean(
                (target - target.mean()) ** 2
            )
        )

        skill = 1.0 - rmse / baseline_rmse

        all_skill.append(skill)

    if idx % 50 == 0:

        print(
            f"{idx}/{len(dataset)} "
            f"MAE={mae:.4f} "
            f"RMSE={rmse:.4f}"
        )

print()
print("=" * 60)

print("Checkpoint:", CHECKPOINT)

print()

print("Mean MAE  :", np.mean(all_mae))
print("Mean RMSE :", np.mean(all_rmse))

if len(all_corr) > 0:
    print("Mean CORR :", np.mean(all_corr))

if len(all_skill) > 0:
    print("Mean SKILL:", np.mean(all_skill))

print()

print("Median MAE :", np.median(all_mae))
print("Median RMSE:", np.median(all_rmse))

print()

print("Best RMSE :", np.min(all_rmse))
print("Worst RMSE:", np.max(all_rmse))

print("=" * 60)