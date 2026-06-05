from datasets.dataset import (
    init_train_valid_datasets_from_config,
    register_dataset
)
from omegaconf import OmegaConf

cfg = OmegaConf.load("conf/config_regression.yaml")

register_dataset(cfg.dataset.type)

dataset, dataset_iterator, val_dataset, val_iterator = \
    init_train_valid_datasets_from_config(
        OmegaConf.to_container(cfg.dataset),
        {},
        batch_size=1,
        validation=True,
        validation_dataset_cfg=OmegaConf.to_container(cfg.validation)
    )

print("Input channels:", dataset.input_channels())
print("Output channels:", dataset.output_channels())

img_clean, img_lr = next(val_iterator)[:2]

print("img_lr:", img_lr.shape)
print("img_clean:", img_clean.shape)