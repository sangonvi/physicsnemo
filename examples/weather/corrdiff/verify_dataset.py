from physicsnemo import Module
import zarr
import torch
from datasets.zarr_dataset import ZarrCorrDiffDataset
import numpy as np
from tqdm import tqdm

CHECKPOINT = (
    "checkpoints/regression/checkpoints_regression/"
    "CorrDiffRegressionUNet.0.100000.mdlus"
)

DATASET_PATH = "zbsvpo23pn bew"
#DATASET_PATH = "/home/sangonvi/Cefet/datasets/corrdiff/train.zarr"

NORMALIZATION = "/home/vsantos/rionowcast/datasets/corrdiff/normalization.npz"
#NORMALIZATION = "/home/sangonvi/Cefet/datasets/corrdiff/normalization.npz"

VALID_INDICES = "/home/vsantos/rionowcast/datasets/corrdiff/valid_index.npy"
#VALID_INDICES = "/home/sangonvi/Cefet/datasets/corrdiff/valid_index.npy"

print("Loading model...")

print("Loading validation dataset...")

dataset = ZarrCorrDiffDataset(
    path=DATASET_PATH,
    normalization_path=NORMALIZATION,
    valid_indices=VALID_INDICES,
    mode="train",
)


model = Module.from_checkpoint(
    "checkpoints/regression/checkpoints_regression/CorrDiffRegressionUNet.0.100000.mdlus"
)

model.cpu()
model.eval()

max_std = 0
max_idx = 0

n_pixels = 0

sum_pixels = 0.0
sum_pixels_sq = 0.0

count_gt0 = 0
count_gt1 = 0
count_gt2 = 0
count_gt5 = 0

target_std_sum = 0.0

norm = np.load(NORMALIZATION)

target_mean = float(norm["target_mean"][0])
target_std = float(norm["target_std"][0])

for i in tqdm(range(len(dataset))):

    target, _ = dataset[i]

    target = target.float()

    target = np.expm1(target)

    values = target.flatten()

    n_pixels += values.numel()

    sum_pixels += values.sum().item()
    sum_pixels_sq += (values ** 2).sum().item()

    count_gt0 += (values > 0).sum().item()
    count_gt1 += (values > 1).sum().item()
    count_gt2 += (values > 2).sum().item()
    count_gt5 += (values > 5).sum().item()

    target_std_sum += values.std().item()

dataset_mean = sum_pixels / n_pixels

dataset_std = (
    sum_pixels_sq / n_pixels
    - dataset_mean ** 2
) ** 0.5

fraction_rain_gt_0 = count_gt0 / n_pixels
fraction_rain_gt_1 = count_gt1 / n_pixels
fraction_rain_gt_2 = count_gt2 / n_pixels
fraction_rain_gt_5 = count_gt5 / n_pixels

mean_target_std = target_std_sum / len(dataset)

print(f"dataset_mean        = {dataset_mean:.6f}")
print(f"dataset_std         = {dataset_std:.6f}")
print(f"fraction_rain_gt_0  = {fraction_rain_gt_0:.6f}")
print(f"fraction_rain_gt_1  = {fraction_rain_gt_1:.6f}")
print(f"fraction_rain_gt_2  = {fraction_rain_gt_2:.6f}")
print(f"fraction_rain_gt_5  = {fraction_rain_gt_5:.6f}")
print(f"mean_target_std     = {mean_target_std:.6f}")
