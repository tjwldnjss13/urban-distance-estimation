
from torchvision.models.resnet import resnet50


if __name__ == '__main__':
    from torchsummary import summary
    model = resnet50()