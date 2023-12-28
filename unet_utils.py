import torch
from torch import nn

class DoubleConvLayer(nn.Module):
    """
    Implementation of the Double Convolutional Layer part of the U-NET Architecture.

    It consists of two convolutional layers:
    - First: in_channels -> out_channels with kernel size = 3
    - Second: out_channels -> out_channels with kernel size = 3

    Parameters
    ----------
    in_channels : int
        No of channels in the input

    out_channels : int
        No of channels in the output
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
        )
    def forward(self, x):
        return self.double_conv(x)
    

class DownSampleLayer(nn.Module):
    """
    Implementation of the Downsampling Layer part of the U-NET Architecture.

    It consists of a convolutional layer with a max pool layer:
    - First: in_channels -> out_channels with kernel size = 3
    - Second: maxpool layer with kernel_size = 2 and stride = 2

    Parameters
    ----------
    in_channels : int
        No of channels in the input

    out_channels : int
        No of channels in the output
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConvLayer(in_channels, out_channels)
        self.pooling = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        down = self.double_conv(x)
        p = self.pooling(down)

        return down, p
    
class UpSampleLayer(nn.Module):
    """
    Implementation of the Upsampling Layer part of the U-NET Architecture.

    It consists of a convolutional layer with a DoubleConvLayer:
    - First: in_channels -> in_channels // 2 with kernel size = 2 and stride = 2
    - Second: DoubleConvLayer with in_channels -> out_channels

    Parameters
    ----------
    in_channels : int
        No of channels in the input

    out_channels : int
        No of channels in the output
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
        self.double_conv = DoubleConvLayer(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], 1) # Output from DownSampleLayer is concatenated here.
        x = self.double_conv(x)
        return x
    
