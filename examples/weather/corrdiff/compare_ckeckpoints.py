import os
import re
import numpy as np
import pandas as pd
import torch

from physicsnemo import Module
from datasets.zarr_dataset import ZarrCorrDiffDataset


DEVICE = "cpu"

CHECKPOINT_DIR = (
    "checkpoints/regression/checkpoints_regression"
)

LOG_FILE = (
    "train.log"
)

DATASET_PATH = (
    "/home/vsantos/rionowcast/datasets/corrdiff/train.zarr"
)

NORMALIZATION = (
    "/home/vsantos/rionowcast/datasets/corrdiff/normalization.npz"
)

VALID_INDICES = (
    "/home/vsantos/rionowcast/datasets/corrdiff/valid_index.npy"
)

# -------------------------------------------------------
# DATASET
# -------------------------------------------------------

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

# -------------------------------------------------------
# PARSE VALIDATION LOSSES
# -------------------------------------------------------

validation_losses = {}

if os.path.exists(LOG_FILE):

    print("Reading log file:", LOG_FILE)

    pattern = re.compile(
        r"VALIDATION: samples=(\d+)\s+validation_loss=([0-9eE\.\-]+)"
    )

    with open(LOG_FILE, "r") as f:

        for line in f:

            m = pattern.search(line)

            if m:

                samples = int(m.group(1))
                loss = float(m.group(2))

                validation_losses[samples] = loss

    print(
        "Validation entries:",
        len(validation_losses)
    )

else:

    print(
        "WARNING: log file not found."
    )

# -------------------------------------------------------
# FIND CHECKPOINTS
# -------------------------------------------------------

checkpoint_files = []

for f in os.listdir(CHECKPOINT_DIR):

    if not f.endswith(".mdlus"):
        continue

    m = re.search(
        r"\.(\d+)\.mdlus$",
        f
    )

    if m:

        checkpoint_files.append(
            (
                int(m.group(1)),
                os.path.join(
                    CHECKPOINT_DIR,
                    f
                )
            )
        )

checkpoint_files.sort(
    key=lambda x: x[0]
)

print()

print("Found checkpoints:")

for step, path in checkpoint_files:

    print(
        step,
        os.path.basename(path)
    )

# -------------------------------------------------------
# EVALUATION
# -------------------------------------------------------

results = []

for step, checkpoint_path in checkpoint_files:

    print()
    print("=" * 80)
    print("Checkpoint:", step)
    print("=" * 80)

    model = Module.from_checkpoint(
        checkpoint_path
    )

    model.eval()
    model.to(DEVICE)

    all_mae = []
    all_rmse = []
    all_corr = []
    all_skill = []

    for idx in range(len(dataset)):

        target, era5 = dataset[idx]

        with torch.no_grad():

            pred = model(
                x=torch.zeros_like(target)
                .unsqueeze(0)
                .to(DEVICE),

                img_lr=era5
                .unsqueeze(0)
                .to(DEVICE)
            )

        target = target.numpy().squeeze()
        pred = pred.cpu().numpy().squeeze()

        # --------------------------
        # denormalization
        # --------------------------

        target = (
            target * target_std
            + target_mean
        )

        pred = (
            pred * target_std
            + target_mean
        )

        mae = np.mean(
            np.abs(pred - target)
        )

        rmse = np.sqrt(
            np.mean(
                (pred - target) ** 2
            )
        )

        all_mae.append(mae)
        all_rmse.append(rmse)

        if (
            target.std() > 1e-6
            and pred.std() > 1e-6
        ):

            corr = np.corrcoef(
                pred.flatten(),
                target.flatten()
            )[0, 1]

            all_corr.append(corr)

            baseline_rmse = np.sqrt(
                np.mean(
                    (
                        target
                        - target.mean()
                    ) ** 2
                )
            )

            skill = (
                1.0
                - rmse / baseline_rmse
            )

            all_skill.append(skill)

    # ---------------------------------------------------
    # validation loss
    # ---------------------------------------------------

    matching_losses = [
        loss
        for samples, loss
        in validation_losses.items()
        if samples <= step
    ]

    if len(matching_losses) > 0:

        validation_last = matching_losses[-1]

        validation_min = np.min(
            matching_losses
        )

        validation_mean = np.mean(
            matching_losses
        )

    else:

        validation_last = np.nan
        validation_min = np.nan
        validation_mean = np.nan

    result = {

        "checkpoint": step,

        "validation_last":
            validation_last,

        "validation_min":
            validation_min,

        "validation_mean":
            validation_mean,

        "mean_mae":
            np.mean(all_mae),

        "mean_rmse":
            np.mean(all_rmse),

        "median_rmse":
            np.median(all_rmse),

        "best_rmse":
            np.min(all_rmse),

        "worst_rmse":
            np.max(all_rmse),

        "mean_corr":
            np.mean(all_corr)
            if len(all_corr)
            else np.nan,

        "mean_skill":
            np.mean(all_skill)
            if len(all_skill)
            else np.nan,
    }

    results.append(result)

    print()

    for k, v in result.items():

        print(
            f"{k:20s}: {v}"
        )

# -------------------------------------------------------
# FINAL TABLE
# -------------------------------------------------------

df = pd.DataFrame(results)

print()
print("=" * 100)
print("SORT BY RMSE")
print("=" * 100)

print(
    df.sort_values(
        "mean_rmse"
    )
)

print()
print("=" * 100)
print("SORT BY CORRELATION")
print("=" * 100)

print(
    df.sort_values(
        "mean_corr",
        ascending=False
    )
)

print()
print("=" * 100)
print("SORT BY VALIDATION LOSS")
print("=" * 100)

print(
    df.sort_values(
        "validation_last"
    )
)

# -------------------------------------------------------
# SAVE CSV
# -------------------------------------------------------

csv_name = (
    "checkpoint_comparison.csv"
)

df.to_csv(
    csv_name,
    index=False
)

print()
print("Saved:", csv_name)

# -------------------------------------------------------
# BEST CHECKPOINTS
# -------------------------------------------------------

best_rmse = df.loc[
    df["mean_rmse"].idxmin()
]

best_corr = df.loc[
    df["mean_corr"].idxmax()
]

print()
print("=" * 100)

print("BEST RMSE")
print(best_rmse)

print()
print("BEST CORRELATION")
print(best_corr)