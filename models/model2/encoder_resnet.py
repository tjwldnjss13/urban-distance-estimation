import torch.nn as nn

from torchvision.models import resnet50

class ResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.layer1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        self.layer2 = nn.Sequential(self.resnet.maxpool, self.resnet.layer1)
        self.layer3 = self.resnet.layer2
        self.layer4 = self.resnet.layer3
        self.layer5 = self.resnet.layer4

        del self.resnet

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        return x1, x2, x3, x4, x5


if __name__ == '__main__':
    from torchsummary import summary
    model = ResnetEncoder().cuda()
    summary(model, (3, 256, 512))