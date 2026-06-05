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

target, era5 = dataset[0]

print(target.shape)
print(era5.shape)

with torch.no_grad():

    pred = model(
        x=torch.zeros_like(target).unsqueeze(0),
        img_lr=era5.unsqueeze(0)
    )

print(pred.shape)
'''
with torch.no_grad():
    pred = model(era5.unsqueeze(0))

print("pred:", pred.shape)
print("pred min:", pred.min())
print("pred max:", pred.max())

'''