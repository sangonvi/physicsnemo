import torch
import numpy as np
import matplotlib.pyplot as plt

from physicsnemo import Module

from datasets.zarr_dataset import ZarrCorrDiffDataset


DEVICE = "cpu"


print("Loading model...")

model = Module.from_checkpoint(
    "checkpoints/regression/checkpoints_regression/CorrDiffRegressionUNet.0.100000.mdlus"
)

model.eval()


print("Loading dataset...")

dataset = ZarrCorrDiffDataset(
    path="/home/vsantos/rionowcast/datasets/corrdiff/train.zarr",
    normalization_path="/home/vsantos/rionowcast/datasets/corrdiff/normalization.npz",
    valid_indices="/home/vsantos/rionowcast/datasets/corrdiff/valid_index.npy",
    mode="valid"
)

norm = np.load(
    "/home/vsantos/rionowcast/datasets/corrdiff/normalization.npz"
)

target_mean = norm["target_mean"]
target_std = norm["target_std"]


rmse_list = []


for idx in range(len(dataset)):

    target, era5 = dataset[idx]

    with torch.no_grad():

        pred = model(
            x=torch.zeros_like(target).unsqueeze(0),
            img_lr=era5.unsqueeze(0)
        )

    pred = pred.squeeze().cpu().numpy()
    target = target.numpy()

    # desnormalização

    pred = pred * target_std[0] + target_mean[0]
    target = target * target_std[0] + target_mean[0]

    rmse = np.sqrt(
        np.mean(
            (pred - target) ** 2
        )
    )

    rmse_list.append(rmse)

    if idx < 5:

        plt.figure(figsize=(12,4))

        plt.subplot(131)
        plt.imshow(target)
        plt.title("Radar Real")

        plt.subplot(132)
        plt.imshow(pred)
        plt.title("Predição")

        plt.subplot(133)
        plt.imshow(pred - target)
        plt.title("Erro")

        plt.savefig(f"sample_{idx}.png")
        plt.close()

    if idx % 100 == 0:
        print(idx, rmse)

print()
print("RMSE médio:", np.mean(rmse_list))
print("RMSE std:", np.std(rmse_list))