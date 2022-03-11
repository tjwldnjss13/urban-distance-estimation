import torch
import torch.nn as nn

from models.model6.conv import Conv, Upconv
from models.model6.attention_module import CBAM


class ResidualBlock13(nn.Module):
    def __init__(self, in_channels, activation_last=True):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, in_channels//2, 1, 1, 0),
            Conv(in_channels//2, in_channels, 3, 1, 1, use_activation=False)
        )
        self.elu = nn.ELU(inplace=True) if activation_last else nn.Sequential()

    def forward(self, x):
        x_conv = self.conv(x)
        x_add = self.elu(x_conv + x)

        return x_add


def residual_13_layers(in_channels, num_layer, activation_last=True):
    return nn.Sequential(
        *[ResidualBlock131(in_channels) for _ in range(num_layer-1)],
        ResidualBlock131(in_channels, activation_last)
    )


class ResidualBlock131(nn.Module):
    def __init__(self, in_channels, activation_last=True):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, in_channels//4, 1, 1, 0),
            Conv(in_channels//4, in_channels//4, 3, 1, 1),
            Conv(in_channels//4, in_channels, 1, 1, 0, use_activation=False)
        )
        self.elu = nn.ELU(inplace=True) if activation_last else nn.Sequential()

    def forward(self, x):
        x_conv = self.conv(x)
        x_add = self.elu(x_conv + x)

        return x_add


def residual_131_layers(in_channels, num_layer, activation_last=True):
    return nn.Sequential(
        *[ResidualBlock131(in_channels) for _ in range(num_layer-1)],
        ResidualBlock131(in_channels, activation_last)
    )


class SRBlock2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = residual_13_layers(in_channels, 3)
        self.conv2 = nn.Sequential(
            Conv(in_channels, 2*in_channels, 3, 2, 1),
            residual_13_layers(2*in_channels, 3),
            Upconv(2*in_channels, in_channels)
        )
        self.conv_last = Conv(2*in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_cat = self.conv_last(torch.cat([x1, x2], dim=1))

        return x_cat


class SRBlock3(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = residual_13_layers(in_channels, 3)
        self.conv2 = nn.Sequential(
            Conv(in_channels, 2*in_channels, 3, 2, 1),
            residual_13_layers(2*in_channels, 3),
            Upconv(2*in_channels, in_channels)
        )
        self.conv3 = nn.Sequential(
            Conv(in_channels, 2*in_channels, 3, 2, 1),
            Conv(2*in_channels, 4*in_channels, 3, 2, 1),
            residual_13_layers(4*in_channels, 3),
            Upconv(4*in_channels, 2*in_channels),
            Upconv(2*in_channels, in_channels)
        )
        self.conv_last = Conv(3*in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_cat = self.conv_last(torch.cat([x1, x2, x3], dim=1))

        return x_cat


class SRDarkNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv(3, 32, 3, 1, 1),
            CBAM(32)
        )
        self.conv2 = nn.Sequential(
            Conv(32, 64, 3, 2, 1),
            residual_131_layers(64, 1),
            CBAM(64)
        )
        self.conv3 = nn.Sequential(
            Conv(64, 128, 3, 2, 1),
            residual_131_layers(128, 2),
            CBAM(128),
        )
        self.conv4 = nn.Sequential(
            Conv(128, 256, 3, 2, 1),
            SRBlock3(256),
            CBAM(256)
        )
        self.conv5 = nn.Sequential(
            Conv(256, 512, 3, 2, 1),
            SRBlock2(512),
            SRBlock2(512),
            CBAM(512)
        )
        self.conv6 = nn.Sequential(
            Conv(512, 1024, 3, 2, 1),
            residual_13_layers(1024, 4)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        return x1, x2, x3, x4, x5, x6


if __name__ == '__main__':
    from torchsummary import summary
    model = SRDarkNet().cuda()
    summary(model, (3, 256, 512))