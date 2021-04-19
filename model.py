import torch
import torch.nn as nn

# We will use this architecture later to implement the CNNblocks via pytorch

architecture = [
    (32, 3, 1),            # Conv2d (32filters , kernelSize 3, Stride 1)
    (64, 3, 2),            # Conv2d (32filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 1],  # Residual block repeated "once"
    (128, 3, 2),           # Conv2d (128filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 2],  # Residual block repeated "twice"
    (256, 3, 2),           # Conv2d (256filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 8],  # Residual block repeated "8 times"
    (512, 3, 2),           # Conv2d (512filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 8],  # Residual block repeated "8 times"
    (1024, 3, 2),          # Conv2d (1024filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 4],  # Residual block repeated "8 times"
    (512, 1, 1),           # Conv2d (512filters , kernelSize 1, Stride 1)
    (1024, 3, 1),          # Conv2d (1024filters , kernelSize 3, Stride 1)
    "ScalePrediction",     # Scale prediction block and computing the yolo loss
    (256, 1, 1),           # Conv2d (256filters , kernelSize 1, Stride 1)
    "Upsampling",          # Upsampling the feature map and concatenating with a previous layer
    (256, 1, 1),           # Conv2d (256filters , kernelSize 1, Stride 1)
    (512, 3, 1),           # Conv2d (512filters , kernelSize 3, Stride 1)
    "ScalePrediction",     # Scale prediction block and computing the yolo loss
    (128, 1, 1),           # Conv2d (128filters , kernelSize 1, Stride 1)
    "Upsampling",          # Upsampling the feature map and concatenating with a previous layer
    (128, 1, 1),           # Conv2d (128filters , kernelSize 1, Stride 1)
    (256, 3, 1),           # Conv2d (128filters , kernelSize 3, Stride 1)
    "ScalePrediction",     # Scale prediction block and computing the yolo loss
]
