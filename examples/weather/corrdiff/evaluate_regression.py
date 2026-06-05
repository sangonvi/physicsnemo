import numpy as np
import torch
import matplotlib.pyplot as plt

from physicsnemo import Module
from datasets.zarr_dataset import ZarrCorrDiffDataset


DEVICE = "cpu"


print("Loading model...")

model = Module.from_checkpoint(
    "checkpoints/regression/checkpoints_regression/CorrDiffRegressionUNet.0.100000.mdlus"
)

model.eval()
model.to(DEVICE)


print("Loading dataset...")

dataset = ZarrCorrDiffDataset(
    path="/home/vsantos/rionowcast/datasets/corrdiff/train.zarr",
    normalization_path="/home/vsantos/rionowcast/datasets/corrdiff/normalization.npz",
    valid_indices="/home/vsantos/rionowcast/datasets/corrdiff/valid_index.npy",
    mode="valid",
)

# --------------------------------------------------
# sample
# --------------------------------------------------

sample_id = 464

target, era5 = dataset[sample_id]

print("target shape:", target.shape)
print("era5 shape:", era5.shape)

# --------------------------------------------------
# inference
# --------------------------------------------------

with torch.no_grad():

    pred = model(
        x=torch.zeros_like(target).unsqueeze(0).to(DEVICE),
        img_lr=era5.unsqueeze(0).to(DEVICE)
    )

pred = pred.cpu()

print("prediction shape:", pred.shape)

# --------------------------------------------------
# denormalization
# --------------------------------------------------

norm = np.load(
    "/home/vsantos/rionowcast/datasets/corrdiff/normalization.npz"
)

target_mean = norm["target_mean"][0]
target_std = norm["target_std"][0]

target = target.numpy()
pred = pred.numpy()

target = target.squeeze(0)
pred = pred.squeeze()

print("Unique values pred:",
      len(np.unique(pred)))

print("Unique values target:",
      len(np.unique(target)))

target = target * target_std + target_mean
pred = pred * target_std + target_mean

print("target min/max/std:",
      target.min(),
      target.max(),
      target.std())

print("pred min/max/std:",
      pred.min(),
      pred.max(),
      pred.std())

print("pred mean :", pred.mean())
print("target mean :", target.mean())
print("pred std  :", pred.std())
print("target std  :", target.std())

# --------------------------------------------------
# metrics
# --------------------------------------------------

mse = np.mean((pred - target) ** 2)

mae = np.mean(np.abs(pred - target))

rmse = np.sqrt(mse)

corr = np.corrcoef(
    pred.flatten(),
    target.flatten()
)[0, 1]

print()
print("MAE :", mae)
print("RMSE:", rmse)
print("CORR:", corr)

# --------------------------------------------------
# plots
# --------------------------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(target)
plt.title("Radar Truth")
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(pred)
plt.title("Regression Prediction")
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(pred - target)
plt.title("Error")
plt.colorbar()

plt.tight_layout()

plt.savefig(
    f"regression_sample_{sample_id}.png",
    dpi=150
)

plt.show()