from physicsnemo import Module
import torch

model = Module.from_checkpoint(
    "checkpoints/regression/checkpoints_regression/CorrDiffRegressionUNet.0.100000.mdlus"
)

model.cpu()
model.eval()

target, era5 = dataset[0]

with torch.no_grad():
    pred = model(era5.unsqueeze(0))

print("target:", target.shape)
print("era5:", era5.shape)
print("pred:", pred.shape)
print("pred min:", pred.min())
print("pred max:", pred.max())