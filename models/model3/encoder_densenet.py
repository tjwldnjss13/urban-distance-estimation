import torch
import torch.nn as nn

from models.model3.conv import Conv


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels, 4*growth_rate, 1, 1, 0),
            nn.BatchNorm2d(4*growth_rate),
            nn.ELU(inplace=True),
            nn.Conv2d(4*growth_rate, growth_rate, 3, 1, 1)
        )

    def forward(self, x):
        x = torch.cat([x, self.layers(x)], dim=1)

        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layer):
        super().__init__()
        self.num_layer = num_layer
        self.conv_list = nn.Sequential(
            *[BottleneckLayer(in_channels+growth_rate*i, growth_rate) for i in range(num_layer)]
        )

    def forward(self, x):
        for n in range(self.num_layer):
            x = self.conv_list[n](x)

        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.layers(x)


class CrossStitchUnit(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.unit = torch.ones(size=(2*size, 2*size), requires_grad=True).cuda() / size

    def forward(self, x1, x2):
        x1_shape = x1.shape
        x2_shape = x2.shape

        x1 = x1.reshape(x1.shape[0], x1.shape[1], -1)
        x2 = x2.reshape(x2.shape[0], x2.shape[1], -1)
        x = torch.cat([x1, x2], dim=-1)

        x = torch.matmul(x, self.unit)
        x1 = x[..., :x1.shape[-1]]
        x2 = x[..., x1.shape[-1]:]

        x1 = x1.reshape(x1_shape)
        x2 = x2.reshape(x2_shape)

        return x1, x2


class DenseNet(nn.Module):
    def __init__(self, growth_rate, reduce_rate):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)

        in_channels = growth_rate
        self.conv1 = Conv(3, in_channels, 7, 2, 3)

        # ------------------------------- Detector ----------------------------------
        num_layer = 6
        self.dense_block1_detector = DenseBlock(in_channels, growth_rate, num_layer)
        in_channels += num_layer * growth_rate
        out_channels = int(in_channels * reduce_rate)
        self.transition1_detector = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        num_layer = 12
        self.dense_block2_detector = DenseBlock(in_channels, growth_rate, num_layer)
        in_channels += num_layer * growth_rate
        out_channels = int(in_channels * reduce_rate)
        self.transition2_detector = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        num_layer = 48
        self.dense_block3_detector = DenseBlock(in_channels, growth_rate, num_layer)
        in_channels += num_layer * growth_rate
        out_channels = int(in_channels * reduce_rate)
        self.transition3_detector = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        num_layer = 32
        self.dense_block4_detector = DenseBlock(in_channels, growth_rate, num_layer)
        # ----------------------------------------------------------------------------
        # -------------------------------- Depth -------------------------------------
        in_channels = growth_rate

        num_layer = 6
        self.dense_block1_depth = DenseBlock(in_channels, growth_rate, num_layer)
        in_channels += num_layer * growth_rate
        out_channels = int(in_channels * reduce_rate)
        self.transition1_depth = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        num_layer = 12
        self.dense_block2_depth = DenseBlock(in_channels, growth_rate, num_layer)
        in_channels += num_layer * growth_rate
        out_channels = int(in_channels * reduce_rate)
        self.transition2_depth = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        num_layer = 48
        self.dense_block3_depth = DenseBlock(in_channels, growth_rate, num_layer)
        in_channels += num_layer * growth_rate
        out_channels = int(in_channels * reduce_rate)
        self.transition3_depth = TransitionLayer(in_channels, out_channels)
        in_channels = out_channels

        num_layer = 32
        self.dense_block4_depth = DenseBlock(in_channels, growth_rate, num_layer)

        self.iconv1_detector = nn.Conv2d(248, 256, 1, 1, 0)
        self.iconv2_detector = nn.Conv2d(892, 512, 1, 1, 0)
        self.iconv3_detector = nn.Conv2d(1916, 1024, 1, 1, 0)

        self.iconv1_depth = nn.Conv2d(32, 64, 1, 1, 0)
        self.iconv2_depth = nn.Conv2d(112, 128, 1, 1, 0)
        self.iconv3_depth = nn.Conv2d(248, 256, 1, 1, 0)
        self.iconv4_depth = nn.Conv2d(892, 512, 1, 1, 0)
        self.iconv5_depth = nn.Conv2d(1916, 1024, 1, 1, 0)

        self.cross_stitch_unit1 = CrossStitchUnit(32 * 64)
        self.cross_stitch_unit2 = CrossStitchUnit(16 * 32)
        self.cross_stitch_unit3 = CrossStitchUnit(8 * 16)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2_detector = self.dense_block1_detector(x2)
        x2_depth = self.dense_block1_depth(x2)
        x2_detector = self.transition1_detector(x2_detector)
        x2_depth = self.transition1_depth(x2_depth)

        x3_detector = self.avgpool(x2_detector)
        x3_depth = self.avgpool(x2_depth)
        x3_detector, x3_depth = self.cross_stitch_unit1(x3_detector, x3_depth)
        x3_detector = self.dense_block2_detector(x3_detector)
        x3_depth = self.dense_block2_depth(x3_depth)
        x3_detector = self.transition2_detector(x3_detector)
        x3_depth = self.transition2_depth(x3_depth)

        x4_detector = self.avgpool(x3_detector)
        x4_depth = self.avgpool(x3_depth)
        x4_detector, x4_depth = self.cross_stitch_unit2(x4_detector, x4_depth)
        x4_detector = self.dense_block3_detector(x4_detector)
        x4_depth = self.dense_block3_depth(x4_depth)
        x4_detector = self.transition3_detector(x4_detector)
        x4_depth = self.transition3_depth(x4_depth)

        x5_detector = self.avgpool(x4_detector)
        x5_depth = self.avgpool(x4_depth)
        x5_detector, x5_depth = self.cross_stitch_unit3(x5_detector, x5_depth)
        x5_detector = self.dense_block4_detector(x5_detector)
        x5_depth = self.dense_block4_depth(x5_depth)

        detector1 = self.iconv1_detector(x3_detector)
        detector2 = self.iconv2_detector(x4_detector)
        detector3 = self.iconv3_detector(x5_detector)

        depth1 = self.iconv1_depth(x1)
        depth2 = self.iconv2_depth(x2_depth)
        depth3 = self.iconv3_depth(x3_depth)
        depth4 = self.iconv4_depth(x4_depth)
        depth5 = self.iconv5_depth(x5_depth)

        feat_detector = [detector1, detector2, detector3]
        feat_depth = [depth1, depth2, depth3, depth4, depth5]

        return feat_detector, feat_depth


if __name__ == '__main__':
    from torchsummary import summary
    model = DenseNet(32, .5).cuda()
    summary(model, (3, 256, 512))












