import torch.nn as nn

from models.model3.conv import Conv, ResidualBlock
from models.model3.attention_module import CBAM


class MyDarknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 32, 3, 1, 1)
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.conv3 = ResidualBlock(64)
        self.conv4 = Conv(64, 128, 3, 2, 1)
        self.conv5 = nn.Sequential(
            # *[ResidualBlock(128) for _ in range(2)]
            *[ResidualBlock(128) for _ in range(4)]
        )
        self.conv6 = Conv(128, 256, 3, 2, 1)
        self.conv7 = nn.Sequential(
            # *[ResidualBlock(256) for _ in range(8)]
            *[ResidualBlock(256) for _ in range(16)]
        )
        self.conv8 = Conv(256, 512, 3, 2, 1)
        self.conv9 = nn.Sequential(
            # *[ResidualBlock(512) for _ in range(8)]
            *[ResidualBlock(512) for _ in range(16)]
        )
        self.conv10 = Conv(512, 1024, 3, 2, 1)
        self.conv11 = nn.Sequential(
            # *[ResidualBlock(1024) for _ in range(4)]
            *[ResidualBlock(1024) for _ in range(8)]
        )

        self.residual1 = Conv(32, 64, 1, 2, 0)
        self.residual2 = Conv(64, 128, 1, 2, 0)
        self.residual3 = Conv(128, 256, 1, 2, 0)
        self.residual4 = Conv(256, 512, 1, 2, 0)
        self.residual5 = Conv(512, 1024, 1, 2, 0)

        self.cbam1 = CBAM(32, 1 / 16)
        self.cbam2 = CBAM(64, 1 / 16)
        self.cbam3 = CBAM(128, 1 / 16)
        self.cbam4 = CBAM(256, 1 / 16)
        self.cbam5 = CBAM(512, 1 / 16)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.cbam1(x)
        x_skip = self.residual1(x)

        x1 = self.conv2(x)
        x1 = self.conv3(x1)
        x1_temp = x1 + x_skip
        x1 = self.cbam2(x1_temp)
        x1_skip = self.residual2(x1)

        x2 = self.conv4(x1)
        x2 = self.conv5(x2)
        x2_temp = x2 + x1_skip
        x2 = self.cbam3(x2_temp)
        x2_skip = self.residual3(x2)

        x3 = self.conv6(x2)
        x3 = self.conv7(x3)
        x3_temp = x3 + x2_skip
        x3 = self.cbam4(x3_temp)
        x3_skip = self.residual4(x3)

        x4 = self.conv8(x3)
        x4 = self.conv9(x4)
        x4_temp = x4 + x3_skip
        x4 = self.cbam5(x4_temp)
        x4_skip = self.residual5(x4)

        x5 = self.conv10(x4)
        x5_temp = self.conv11(x5)
        x5 = x5_temp + x4_skip

        return x1, x2, x3, x4, x5


if __name__ == '__main__':
    from torchsummary import summary
    model = Darknet53().cuda()
    summary(model, (3, 256, 512))