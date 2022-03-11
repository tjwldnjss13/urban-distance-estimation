import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model8.conv import Conv, UpconvBilinear


class DisparityPrediction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = Conv(in_channels, 2, 1, 1, 0, use_activation=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x) * .3

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.upconv5 = nn.Sequential(
            UpconvBilinear(1024, 512),
            Conv(512, 512)
        )
        self.upconv4 = nn.Sequential(
            UpconvBilinear(512, 256),
            Conv(256, 256)
        )
        self.upconv3 = nn.Sequential(
            UpconvBilinear(256, 128),
            Conv(128, 128)
        )
        self.upconv2 = nn.Sequential(
            UpconvBilinear(128, 64),
            Conv(64, 64)
        )
        self.upconv1 = nn.Sequential(
            UpconvBilinear(64, 32),
            Conv(32, 32)
        )

        self.iconv5 = Conv(512+512, 512)
        self.iconv4 = Conv(256+256, 256)
        self.iconv3 = Conv(128+128+2, 128)
        self.iconv2 = Conv(64+64+2, 64)
        self.iconv1 = Conv(32+32+2, 32)

        self.disp4 = DisparityPrediction(256)
        self.disp3 = DisparityPrediction(128)
        self.disp2 = DisparityPrediction(64)
        self.disp1 = DisparityPrediction(32)

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

    def forward(self, skip1, skip2, skip3, skip4, skip5, x):
        up5 = self.upconv5(x)
        cat5 = torch.cat([up5, skip5], dim=1)
        i5 = self.iconv5(cat5)

        up4 = self.upconv4(i5)
        cat4 = torch.cat([up4, skip4], dim=1)
        i4 = self.iconv4(cat4)
        disp4 = self.disp4(i4)
        updisp4 = F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=True)

        up3 = self.upconv3(i4)
        cat3 = torch.cat([up3, skip3, updisp4], dim=1)
        i3 = self.iconv3(cat3)
        disp3 = self.disp3(i3)
        updisp3 = F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=True)

        up2 = self.upconv2(i3)
        cat2 = torch.cat([up2, skip2, updisp3], dim=1)
        i2 = self.iconv2(cat2)
        disp2 = self.disp2(i2)
        updisp2 = F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=True)

        up1 = self.upconv1(i2)
        cat1 = torch.cat([up1, skip1, updisp2], dim=1)
        i1 = self.iconv1(cat1)
        disp1 = self.disp1(i1)

        return [disp1, disp2, disp3, disp4]


if __name__ == '__main__':
    from torchsummary import summary
    model = Decoder().cuda()
    skip1 = torch.ones(2, 32, 256, 512).cuda()
    skip2 = torch.ones(2, 64, 128, 256).cuda()
    skip3 = torch.ones(2, 128, 64, 128).cuda()
    skip4 = torch.ones(2, 256, 32, 64).cuda()
    skip5 = torch.ones(2, 512, 16, 32).cuda()
    x = torch.ones(2, 1024, 8, 16).cuda()

    out = model(skip1, skip2, skip3, skip4, skip5, x)






