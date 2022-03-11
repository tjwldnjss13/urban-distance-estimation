import torch
import torch.nn as nn

from models.model4.conv import Conv
from models.model4.attention_module import CBAM


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, in_channels//2, 1, 1, 0),
            Conv(in_channels//2, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        x_conv = self.conv(x)
        return x_conv + x


class DarkNet53(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            Conv(3, 32, 3, 1, 1),
            CBAM(32, 1/16)
        )
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.conv3 = nn.Sequential(
            self._make_residual_block(64, 1),
            CBAM(64, 1/16),
            Conv(64, 128, 3, 2, 1)
        )
        self.conv4 = nn.Sequential(
            self._make_residual_block(128, 2),
            CBAM(128, 1/16),
            Conv(128, 256, 3, 2, 1)
        )
        self.conv5 = nn.Sequential(
            self._make_residual_block(256, 8),
            CBAM(256, 1/16),
            Conv(256, 512, 3, 2, 1)
        )
        self.conv6 = nn.Sequential(
            self._make_residual_block(512, 8),
            CBAM(512, 1/16),
            Conv(512, 1024, 3, 2, 1),
            self._make_residual_block(1024, 4)
        )

    def _make_residual_block(self, in_channels, num_block):
        return nn.Sequential(*[ResidualBlock(in_channels) for _ in range(num_block)])

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
    model = DarkNet53().cuda()
    summary(model, (3, 224, 448))