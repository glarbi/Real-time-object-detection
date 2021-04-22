import torch
import torch.nn as nn

architecture = [
    (32, 3, 1),  # Conv2d (32filters , kernelSize 3, Stride 1)
    (64, 3, 2),  # Conv2d (64filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 1],  # Residual block repeated "once"
    (128, 3, 2),  # Conv2d (128filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 2],  # Residual block repeated "twice"
    (256, 3, 2),  # Conv2d (256filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 8],  # Residual block repeated "8 times"
    (512, 3, 2),  # Conv2d (512filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 8],  # Residual block repeated "8 times"
    (1024, 3, 2),  # Conv2d (1024filters , kernelSize 3, Stride 2)
    ["ResidualBlock", 4],  # Residual block repeated "8 times", at this point is Darknet53
    (512, 1, 1),  # Conv2d (512filters , kernelSize 1, Stride 1)
    (1024, 3, 1),  # Conv2d (1024filters , kernelSize 3, Stride 1)
    "ScalePrediction",  # Scale prediction block and computing the yolo loss
    (256, 1, 1),  # Conv2d (256filters , kernelSize 1, Stride 1)
    "Upsampling",  # Upsampling the feature map and concatenating with a previous layer
    (256, 1, 1),  # Conv2d (256filters , kernelSize 1, Stride 1)
    (512, 3, 1),  # Conv2d (512filters , kernelSize 3, Stride 1)
    "ScalePrediction",  # Scale prediction block and computing the yolo loss
    (128, 1, 1),  # Conv2d (128filters , kernelSize 1, Stride 1)
    "Upsampling",  # Upsampling the feature map and concatenating with a previous layer
    (128, 1, 1),  # Conv2d (128filters , kernelSize 1, Stride 1)
    (256, 3, 1),  # Conv2d (256filters , kernelSize 3, Stride 1)
    "ScalePrediction",  # Scale prediction block and computing the yolo loss
]


class Convolutional_Block(nn.Module):  # Our CNN Block class
    def __init__(self, in_channels, out_channels, batchNorm=True,
                 **kwargs):  # batchNorm bool : is whether the block is going to use batchNorm
        super().__init__()  # supercall to inherit from nn.module
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not batchNorm,
                              **kwargs)  # if batchNorm is used then bias is unnecessary
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)  # the activation function
        self.use_batchNorm = batchNorm

    def forward(self, x):
        if self.use_batchNorm:  # when Scalepredicting output is conv else Leaky,batch,conv
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class Residual_Block(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):  # num of repeat is sent in from architecture
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):  # loop through numberof repeats
            self.layers += [nn.Sequential(Convolutional_Block(channels, channels // 2, kernel_size=1),
                                          Convolutional_Block(channels // 2, channels, kernel_size=3, padding=1))]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)  # f(x) + x
            else:
                x = layer(x)  # identity
        return x


class Scale_Prediction(nn.Module):  # OUtputs our output
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.predictions = nn.Sequential(
            Convolutional_Block(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            Convolutional_Block(2 * in_channels, (num_classes + 5) * 3, batchNorm=False, kernel_size=1)
            # number of classes + 5, 5 is [P0,x,y,w,h]
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (self.predictions(x)
                .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)
                )


class Yolo(nn.Module):  # our model class
    def __init__(self, in_channels=3, num_classes=20):  # num classes of Pascal VOC
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self.create_layers()

    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, Scale_Prediction):
                outputs.append(layer(x))
                continue  # skipping

            x = layer(x)

            if isinstance(layer, Residual_Block) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def create_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in architecture:
            if isinstance(module, tuple):  # module is a convolutional block
                out_channels, kernel_size, stride = module
                layers.append(Convolutional_Block(in_channels, out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=1 if kernel_size == 3 else 0,
                                                  )
                              )
                in_channels = out_channels
            elif isinstance(module, list):  # module is a residual block
                num_repeats = module[1]  # how many times the residual block is repeated
                layers.append(Residual_Block(in_channels, num_repeats=num_repeats))

            elif isinstance(module, str):
                if module == "ScalePrediction":
                    layers += [
                        Residual_Block(in_channels, use_residual=False, num_repeats=1),
                        Convolutional_Block(in_channels, in_channels // 2, kernel_size=1),
                        Scale_Prediction(in_channels // 2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                elif module == "Upsampling":
                    layers.append(nn.Upsample(scale_factor=2), )
                    in_channels = in_channels * 3  # we should concatenate just after the Upsample
        return layers


##### Testing the model ####
num_classes = 20
IMAGE_SIZE = 416
model = Yolo(num_classes=num_classes)
x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
out = model(x)
assert model(x)[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
assert model(x)[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
assert model(x)[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5)
print("Success!")