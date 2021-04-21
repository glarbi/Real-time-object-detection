import torch
import torch.nn as nn

architecture = [
    (32, 3, 1),            # Conv2d (32filters , kernelSize 3, Stride 1)
    (64, 3, 2),            # Conv2d (64filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 1],  # Residual block repeated "once"
    (128, 3, 2),           # Conv2d (128filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 2],  # Residual block repeated "twice"
    (256, 3, 2),           # Conv2d (256filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 8],  # Residual block repeated "8 times"
    (512, 3, 2),           # Conv2d (512filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 8],  # Residual block repeated "8 times"
    (1024, 3, 2),          # Conv2d (1024filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 4],  # Residual block repeated "8 times", at this point is Darknet53
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
    (256, 3, 1),           # Conv2d (256filters , kernelSize 3, Stride 1)
    "ScalePrediction",     # Scale prediction block and computing the yolo loss
]

class Convolutional_Block(nn.Module): # Our CNN Block class
    def __init__(self, in_channels, out_channels, batchNorm = True, **kwargs): # batchNorm bool : is whether the block is going to use batchNorm
        super().__init__() # supercall to inherit from nn.module
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not batchNorm, **kwargs) # if batchNorm is used then bias is unnecessary
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU() # the activation function
        self.use_batchNorm = batchNorm

    def forward(self, x):
        if self.use_batchNorm: # when Scalepredicting output is conv else Leaky,batch,conv
            return self.leaky(self.batchnorm(self.conv(x)))
        else:
            return self.conv(x)


class Residual_Block(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1): # num of repeat is sent in from architecture
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in num_repeats: # loop through numberof repeats
            self.layers = self.layers + nn.Sequential(Convolutional_Block(channels, channels//2, kernel_size=1),
                                                      Convolutional_Block(channels//2, channels, kernel_size=3, paddings=1))
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x    # f(x) + x
            else :
                x = layer(x)    # identity
        return x

class Scale_Prediction(nn.Module):
    pass # not yet

class Yolo(nn.Module): # our model class
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.layers = self.create_layers()

    def forward(self, x):
        pass # not yet

    def create_layers(self):
        pass # not yet