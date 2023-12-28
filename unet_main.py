import torch
from torch import nn

from unet_utils import DoubleConvLayer, DownSampleLayer, UpSampleLayer

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.downconv1 = DownSampleLayer(in_channels, 64)
        self.downconv2 = DownSampleLayer(64, 128)
        self.downconv3 = DownSampleLayer(128, 256)
        self.downconv4 = DownSampleLayer(256, 512)

        self.bottleneck = DoubleConvLayer(512, 1024)

        self.upconv1 = UpSampleLayer(1024, 512)
        self.upconv2 = UpSampleLayer(512, 256)
        self.upconv3 = UpSampleLayer(256, 128)
        self.upconv4 = UpSampleLayer(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size = 1)

    def forward(self, x):
        down1, p1 = self.downconv1(x)
        down2, p2 = self.downconv2(p1)
        down3, p3 = self.downconv3(p2)
        down4, p4 = self.downconv4(p3)

        bottle = self.bottleneck(p4)

        up1 = self.upconv1(bottle, down4)
        up2 = self.upconv2(up1, down3)
        up3 = self.upconv3(up2, down2)
        up4 = self.upconv4(up3, down1)

        out = self.out(up4)

        return out
    

if __name__ == "__main__":
    double_conv = DoubleConvLayer(256, 256)
    print(double_conv)

    input_image = torch.rand((1, 3, 512, 512))

    model = UNet(3, 10)

    output = model(input_image)
    print(output.size())