import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True, use_activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Sequential()
        self.activation = nn.ReLU(inplace=True) if use_activation else nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, num_repeat, use_bn=False, use_activation=True):
        super().__init__()
        self.convs = nn.Sequential(
            *[nn.Sequential(Conv(in_channels, in_channels, 1, 1, 0, use_bn=use_bn, use_activation=use_activation), Conv(in_channels, in_channels, 3, 1, 1, use_bn=use_bn, use_activation=use_activation)) for _ in range(num_repeat)]
        )

    def forward(self, x):
        return self.convs(x)



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample:
            stride = 2
        else:
            stride = 1
        mid_channels = out_channels // 4
        self.conv1 = Conv(in_channels, mid_channels, 1, stride, 0)
        self.conv2 = Conv(mid_channels, mid_channels, 3, 1, 1)
        self.conv3 = Conv(mid_channels, out_channels, 1, 1, 0, use_activation=False)
        self.conv_skip = Conv(in_channels, out_channels, 1, stride, 0, use_activation=False)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        skip = self.conv_skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += skip
        x = self.activation(x)

        return x


class UpconvBilinear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=False, use_activation=True):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding, use_bn, use_activation)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = ResidualBlock(32, 64, True).cuda()
    upconv = UpconvBilinear(32, 16, 3, 1, 1).cuda()
    summary(upconv, (32, 128, 256))

























