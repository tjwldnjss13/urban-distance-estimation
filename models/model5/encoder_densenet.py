import torch
import torch.nn as nn

from models.model5.conv import Conv


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


# class CrossStitchUnit(nn.Module):
#     def __init__(self, size):
#         super().__init__()
#
#     def forward(self, x1, x2):
#
#         return x1, x2


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


class DenseNet(nn.Module):
    def __init__(self, img_size, growth_rate, reduce_rate):
        super().__init__()
        H, W = img_size
        num_layers = [6, 12, 24, 16]
        in_channels = growth_rate
        out_channels_list = [in_channels]

        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)

        self.conv1 = Conv(3, in_channels, 7, 1, 3)

        # ------------------------------- Depth ----------------------------------
        out_channels = in_channels * 2
        self.block1_depth = Conv(in_channels, out_channels, 3, 1, 1)
        out_channels_list.append(out_channels)

        num_layer = num_layers[0]
        # self.dense_block1_depth = DenseBlock(in_channels, growth_rate, num_layer)
        # in_channels += num_layer * growth_rate
        # out_channels = int(in_channels * reduce_rate)
        # self.transition1_depth = TransitionLayer(in_channels, out_channels)
        # in_channels = out_channels
        # out_channels_list.append(out_channels)
        in_channels_dense = out_channels_list[1]
        in_channels_trans = in_channels_dense + num_layer * growth_rate
        out_channels_trans = int(in_channels_trans * reduce_rate)
        out_channels_list.append(out_channels_trans)
        self.block2_depth = nn.Sequential(
            DenseBlock(in_channels_dense, growth_rate, num_layer),
            TransitionLayer(in_channels_trans, out_channels_trans)
        )

        num_layer = num_layers[1]
        # self.dense_block2_depth = DenseBlock(in_channels, growth_rate, num_layer)
        # in_channels += num_layer * growth_rate
        # out_channels = int(in_channels * reduce_rate)
        # self.transition2_depth = TransitionLayer(in_channels, out_channels)
        # in_channels = out_channels
        # out_channels_list.append(out_channels)
        in_channels_dense = out_channels_trans
        in_channels_trans = in_channels_dense + num_layer * growth_rate
        out_channels_trans = int(in_channels_trans * reduce_rate)
        out_channels_list.append(out_channels_trans)
        self.block3_depth = nn.Sequential(
            DenseBlock(in_channels_dense, growth_rate, num_layer),
            TransitionLayer(in_channels_trans, out_channels_trans)
        )

        num_layer = num_layers[2]
        # self.dense_block3_depth = DenseBlock(in_channels, growth_rate, num_layer)
        # in_channels += num_layer * growth_rate
        # out_channels = int(in_channels * reduce_rate)
        # self.transition3_depth = TransitionLayer(in_channels, out_channels)
        # in_channels = out_channels
        # out_channels_list.append(out_channels)
        in_channels_dense = out_channels_trans
        in_channels_trans = in_channels_dense + num_layer * growth_rate
        out_channels_trans = int(in_channels_trans * reduce_rate)
        out_channels_list.append(out_channels_trans)
        self.block4_depth = nn.Sequential(
            DenseBlock(in_channels_dense, growth_rate, num_layer),
            TransitionLayer(in_channels_trans, out_channels_trans)
        )

        num_layer = num_layers[3]
        # self.dense_block4_depth = DenseBlock(in_channels, growth_rate, num_layer)
        # out_channels = int(in_channels * reduce_rate)
        # self.transition4_depth = TransitionLayer(in_channels, out_channels)
        # out_channels_list.append(out_channels)
        in_channels_dense = out_channels_trans
        out_channels = in_channels_dense + num_layer * growth_rate
        out_channels_list.append(out_channels)
        self.block5_depth = nn.Sequential(
            DenseBlock(in_channels_dense, growth_rate, num_layer),
        )
        # ----------------------------------------------------------------------------
        # -------------------------------- Detector -------------------------------------
        self.block1_detector = nn.Sequential(
            Conv(32, 64, 1, 1, 0),
            self._make_residual_block(64, 1)
        )
        self.block2_detector = nn.Sequential(
            Conv(64, 128, 1, 1, 0),
            self._make_residual_block(128, 2)
        )
        self.block3_detector = nn.Sequential(
            Conv(128, 256, 1, 1, 0),
            self._make_residual_block(256, 8)
        )
        self.block4_detector = nn.Sequential(
            Conv(256, 512, 1, 1, 0),
            self._make_residual_block(512, 8)
        )
        self.block5_detector = nn.Sequential(
            Conv(512, 1024, 1, 1, 0),
            self._make_residual_block(1024, 4)
        )

        self.cross_stitch_unit1 = CrossStitchUnit(H//2**2 * W//2**2)
        self.cross_stitch_unit2 = CrossStitchUnit(H//2**3 * W//2**3)
        self.cross_stitch_unit3 = CrossStitchUnit(H//2**4 * W//2**4)
        self.cross_stitch_unit4 = CrossStitchUnit(H//2**5 * W//2**5)

    def _make_residual_block(self, in_channels, num_block):
        return nn.Sequential(*[ResidualBlock(in_channels) for _ in range(num_block)])

    def forward(self, x):
        x1 = x = self.conv1(x)
        # 32, 2^0

        x = self.maxpool(x)
        x2_detector = self.block1_detector(x)
        x2_depth = self.block1_depth(x)
        # 64, 2^1

        x3_detector = self.avgpool(x2_detector)
        x3_depth = self.avgpool(x2_depth)
        x3_detector, x3_depth = self.cross_stitch_unit1(x3_detector, x3_depth)
        x3_detector = self.block2_detector(x3_detector)
        x3_depth = self.block2_depth(x3_depth)
        # 128, 2^2

        x4_detector = self.avgpool(x3_detector)
        x4_depth = self.avgpool(x3_depth)
        x4_detector, x4_depth = self.cross_stitch_unit2(x4_detector, x4_depth)
        x4_detector = self.block3_detector(x4_detector)
        x4_depth = self.block3_depth(x4_depth)
        # 256, 2^3

        x5_detector = self.avgpool(x4_detector)
        x5_depth = self.avgpool(x4_depth)
        x5_detector, x5_depth = self.cross_stitch_unit3(x5_detector, x5_depth)
        x5_detector = self.block4_detector(x5_detector)
        x5_depth = self.block4_depth(x5_depth)
        # 512, 2^4

        x6_detector = self.avgpool(x5_detector)
        x6_depth = self.avgpool(x5_depth)
        x6_detector, x6_depth = self.cross_stitch_unit4(x6_detector, x6_depth)
        x6_detector = self.block5_detector(x6_detector)
        x6_depth = self.block5_depth(x6_depth)
        # 1024, 2^5

        feat_detector = [x4_detector, x5_detector, x6_detector]
        feat_depth = [x1, x2_depth, x3_depth, x4_depth, x5_depth, x6_depth]

        return feat_detector, feat_depth


if __name__ == '__main__':
    from torchsummary import summary
    h = 256
    w = 2 * h
    model = DenseNet((h, w), 32, .5).cuda()
    summary(model, (3, h, w))












