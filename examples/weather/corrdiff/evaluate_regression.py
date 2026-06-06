import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from physicsnemo import Module
from datasets.zarr_dataset import ZarrCorrDiffDataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "--step",
    type=int,
    required=True
)

parser.add_argument(
    "--sample",
    type=int,
    default=464,
    help="Validation sample id"
)

args = parser.parse_args()

CHECKPOINT = (
    "checkpoints/regression/checkpoints_regression/"
    f"CorrDiffRegressionUNet.0.{args.step}.mdlus"
)

SAMPLE_ID = args.sample

DEVICE = "cpu"

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

target, era5 = dataset[SAMPLE_ID]

print("target shape:", target.shape)
print("era5 shape:", era5.shape)

with torch.no_grad():

    pred = model(
        x=torch.zeros_like(target).unsqueeze(0).to(DEVICE),
        img_lr=era5.unsqueeze(0).to(DEVICE)
    )

pred = pred.cpu()

print("prediction shape:", pred.shape)

norm = np.load(NORMALIZATION)

target_mean = norm["target_mean"][0]
target_std = norm["target_std"][0]

target = target.numpy().squeeze()
pred = pred.numpy().squeeze()

target = target * target_std + target_mean
pred = pred * target_std + target_mean

mse = np.mean((pred - target) ** 2)
mae = np.mean(np.abs(pred - target))
rmse = np.sqrt(mse)

if target.std() > 1e-6 and pred.std() > 1e-6:
    corr = np.corrcoef(
        pred.flatten(),
        target.flatten()
    )[0, 1]
else:
    corr = np.nan

baseline_rmse = np.sqrt(
    np.mean((target - target.mean()) ** 2)
)

skill = 1.0 - rmse / baseline_rmse

print(pred.mean(), pred.std())
print(target.mean(), target.std())
print()
print("MAE   :", mae)
print("RMSE  :", rmse)
print("CORR  :", corr)
print("SKILL :", skill)

vmin = min(target.min(), pred.min())
vmax = max(target.max(), pred.max())

error = pred - target

plt.figure(figsize=(16,5))

plt.subplot(1,3,1)

plt.imshow(
    target,
    vmin=vmin,
    vmax=vmax
)

plt.title("Radar Truth")
plt.colorbar()

plt.subplot(1,3,2)

plt.imshow(
    pred,
    vmin=vmin,
    vmax=vmax
)

plt.title("Prediction")
plt.colorbar()

plt.subplot(1,3,3)

plt.imshow(
    error,
    cmap="RdBu_r",
    vmin=-np.max(np.abs(error)),
    vmax=np.max(np.abs(error))
)

plt.title("Prediction Error")
plt.colorbar()

plt.tight_layout()

output_file = f"regression_sample_{SAMPLE_ID}.png"

plt.savefig(
    output_file,
    dpi=150
)

print()
print("Saved:", output_file)

plt.show()