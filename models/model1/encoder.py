import torch
import torch.nn as nn

from models.model1.conv import Conv, ConvBlock
from models.model1.attention_module import CBAM


class Encoder(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.in_size = in_size

        self.layer1 = Conv(3, 64, 7, 2, 3)
        self.layer2 = nn.Sequential(
            Conv(64, 128, 3, 2, 1),
            ConvBlock(128, 2)
            # ResidualBlock(64, 128, True),
            # *[ResidualBlock(128, 128) for _ in range(2)]
        )
        self.layer3 = nn.Sequential(
            Conv(128, 256, 3, 2, 1),
            ConvBlock(256, 4)
            # ResidualBlock(128, 256, True),
            # *[ResidualBlock(256, 256) for _ in range(3)]
        )
        self.layer4 = nn.Sequential(
            Conv(256, 512, 3, 2, 1),
            ConvBlock(512, 9)
            # ResidualBlock(256, 512, True),
            # *[ResidualBlock(512, 512) for _ in range(22)]
        )
        self.layer5 = nn.Sequential(
            Conv(512, 1024, 3, 2, 1),
            ConvBlock(1024, 4)
            # ResidualBlock(512, 1024, True),
            # *[ResidualBlock(1024, 1024) for _ in range(2)]
        )

        self.cbam1 = CBAM(64, 1/16)
        self.cbam2 = CBAM(128, 1/16)
        self.cbam3 = CBAM(256, 1/16)
        self.cbam4 = CBAM(512, 1/16)

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = self.cbam1(x1)

        x2 = self.layer2(x1)
        x2 = self.cbam2(x2)

        x3 = self.layer3(x2)
        x3 = self.cbam3(x3)

        x4 = self.layer4(x3)
        x4 = self.cbam4(x4)

        x5 = self.layer5(x4)

        return x1, x2, x3, x4, x5


if __name__ == '__main__':
    from torchsummary import summary
    model = Encoder((256, 512)).cuda()
    summary(model, (3, 256, 512))








