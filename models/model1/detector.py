import torch
import torch.nn as nn

from models.model1.conv import Conv


class Detector(nn.Module):
    def __init__(self, feat_size, num_classes):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.feat_size = feat_size
        self.num_classes = num_classes

        self.detector = Conv(1024, (5 + num_classes) * 5, 1, 1, 0, use_activation=False)

        self.anchors = torch.Tensor([[1.0655, 1.6272],
                                     [2.4673, 3.9295],
                                     [4.9840, 6.2226],
                                     [2.9788, 11.6568],
                                     [6.1582, 13.8294]])
        self.anchor_boxes = self._generate_anchor_box()

    def _generate_anchor_box(self):
        """
        Make anchor box the same shape as output's.
        :param anchor_box_sizes: tensor, [# anchor box, (height, width)]
        :param out_size: tuple or list, (height, width)

        :return tensor, [height, width, (cy, cx, h, w) * num bounding box]
        """

        out = torch.zeros(self.feat_size[0], self.feat_size[1], 4 * len(self.anchors)).cuda()
        cy_ones = torch.ones(self.feat_size[0], 1)
        cx_ones = torch.ones(1, self.feat_size[1])
        cy_tensor = torch.zeros(self.feat_size[0], 1)
        cx_tensor = torch.zeros(1, self.feat_size[1])

        for i in range(1, self.feat_size[0]):
            cx_tensor = torch.cat([cx_tensor, cx_ones * i], dim=0)

        for i in range(1, self.feat_size[1]):
            cy_tensor = torch.cat([cy_tensor, cy_ones * i], dim=1)

        ctr_tensor = torch.cat([cy_tensor.unsqueeze(2), cx_tensor.unsqueeze(2)], dim=2)

        for i in range(len(self.anchors)):
            out[:, :, 4 * i:4 * i + 2] = ctr_tensor
            out[:, :, 4 * i + 2] = self.anchors[i, 0]
            out[:, :, 4 * i + 3] = self.anchors[i, 1]

        return out

    def _activate_detector(self, detector):
        detector = detector.reshape(detector.shape[0], -1, 5+self.num_classes)
        out = torch.zeros(detector.shape).to(self.device)
        anchor_boxes = self.anchor_boxes.reshape(-1, 4).repeat(detector.shape[0], 1, 1)
        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax(dim=-1)

        out[..., 0:2] = anchor_boxes[..., 0:2] + sigmoid(detector[..., 0:2])
        out[..., 2:4] = anchor_boxes[..., 2:4] * torch.exp(detector[..., 2:4])
        out[..., 4] = sigmoid(detector[..., 4])
        out[..., 5:] = softmax(detector[..., 5:])

        return out

    def forward(self, x):
        x = self.detector(x)
        x = self._activate_detector(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = Detector((8, 16), 9).cuda()
    summary(model, (1024, 8, 16))












