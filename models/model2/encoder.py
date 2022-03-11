import torch
import torch.nn as nn

from models.model2.conv import Conv, ResidualBlock
from models.model2.attention_module import CBAM


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layer1 = Conv(3, 64, 7, 2, 3)
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 256, True),
            *[ResidualBlock(256, 256) for _ in range(2)]
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(256, 512, True),
            *[ResidualBlock(512, 512) for _ in range(3)]
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(512, 1024, True),
            *[ResidualBlock(1024, 1024) for _ in range(5)]
        )
        self.layer5 = nn.Sequential(
            ResidualBlock(1024, 2048, True),
            *[ResidualBlock(2048, 2048) for _ in range(2)]
        )

        self.cbam1 = CBAM(64, 1/16)
        self.cbam2 = CBAM(256, 1/16)
        self.cbam3 = CBAM(512, 1/16)
        self.cbam4 = CBAM(1024, 1/16)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
    model = Encoder().cuda()
    summary(model, (3, 256, 512))








