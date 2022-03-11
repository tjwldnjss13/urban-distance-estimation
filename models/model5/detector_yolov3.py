import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms

from models.model5.conv import Conv, ConvSet
from utils.pytorch_util import convert_box_from_yxhw_to_xyxy


class YOLOV3Detector(nn.Module):
    def __init__(self, feat_size, num_classes):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.feat_size = feat_size
        self.num_classes = num_classes

        self.train_safe = False

        self.num_box_predict = 3
        self.out_size = [(feat_size[0] * 2 ** i, feat_size[1] * 2 ** i) for i in range(3)]
        self.anchors = torch.Tensor([[0.22, 0.28], [0.48, 0.38], [0.78, 0.9],
                                     [0.15, 0.07], [0.11, 0.15], [0.29, 0.14],
                                     [0.03, 0.02], [0.07, 0.04], [0.06, 0.08]]).to(self.device)
        self.anchor_boxes = self._generate_anchor_boxes()

        self.stage1_conv = ConvSet(1024, 512, 1024)
        self.stage1_conv_skip = Conv(1024, 256, 1, 1, 0)
        self.stage1_detector = nn.Sequential(
            Conv(1024, 256, 1, 1, 0),
            Conv(256, self.num_box_predict*(5+self.num_classes), 1, 1, 0, False, False)
        )

        self.stage2_conv = ConvSet(768, 256, 512)
        self.stage2_conv_skip = Conv(512, 128, 1, 1, 0)
        self.stage2_detector = nn.Sequential(
            Conv(512, 128, 1, 1, 0),
            Conv(128, self.num_box_predict*(5+self.num_classes), 1, 1, 0, False, False)
        )

        self.stage3_conv = ConvSet(384, 128, 256)
        self.stage3_detector = nn.Sequential(
            Conv(256, 64, 1, 1, 0),
            Conv(64, self.num_box_predict*(5+self.num_classes), 1, 1, 0, False, False)
        )

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

    def _generate_anchor_box(self, out_size, anchors):
        """
        Make anchor box the same shape as output's.
        :param anchor_box_sizes: tensor, [# anchor box, (height, width)]
        :param out_size: tuple or list, (height, width)

        :return tensor, [height, width, (cy, cx, h, w) * num bounding box]
        """

        out = torch.zeros(out_size[0], out_size[1], 4 * len(anchors)).to(self.device)

        for i in range(out_size[0]):
            out[i, :, 0:-1:4] = i
        for i in range(out_size[1]):
            out[:, i, 1:-1:4] = i
        for i in range(len(anchors)):
            out[..., 4*i+2] = anchors[i, 0] * out_size[0]
            out[..., 4*i+3] = anchors[i, 1] * out_size[1]

        return out

    def _generate_anchor_boxes(self):
        anchor_box_list = []
        for i in range(len(self.out_size)):
            anchor_box = self._generate_anchor_box(self.out_size[i], self.anchors[3*i:3*(i+1)])
            anchor_box_list.append(anchor_box)

        return anchor_box_list

    def _activate_detector(self, x):
        B, H, W = x.shape[:3]
        x = x.reshape(B, -1, 5 + self.num_classes)
        x = torch.cat([
            torch.sigmoid(x[..., :2]),
            x[..., 2:4],
            torch.sigmoid(x[..., 4:])
        ], dim=-1)
        x = x.reshape(B, H, W, -1)

        return x

    def forward(self, x3, x4, x5):
        """
        :return: detectors, [num batch, height of scale, width of scale, num pred box * (5 + num class)] for each scale
        """

        s1 = self.stage1_conv(x5)
        s1_detector = self.stage1_detector(s1)

        s1_skip = self.stage1_conv_skip(s1)
        s1_skip = F.interpolate(s1_skip, scale_factor=2, mode='bicubic', align_corners=True)

        s2 = torch.cat([s1_skip, x4], dim=1)
        s2 = self.stage2_conv(s2)
        s2_detector = self.stage2_detector(s2)

        s2_skip = self.stage2_conv_skip(s2)
        s2_skip = F.interpolate(s2_skip, scale_factor=2, mode='bicubic', align_corners=True)

        s3 = torch.cat([s2_skip, x3], dim=1)
        s3 = self.stage3_conv(s3)
        s3_detector = self.stage3_detector(s3)

        s1_detector = s1_detector.permute(0, 2, 3, 1)
        s2_detector = s2_detector.permute(0, 2, 3, 1)
        s3_detector = s3_detector.permute(0, 2, 3, 1)

        s1_detector = self._activate_detector(s1_detector)
        s2_detector = self._activate_detector(s2_detector)
        s3_detector = self._activate_detector(s3_detector)

        return s1_detector, s2_detector, s3_detector

    def final_result(self, predict_list, confidence_threshold, nms_threshold):
        box_list = []
        conf_list = []
        cls_list = []

        box_per_cls_list = []
        conf_per_cls_list = []

        for i in range(len(predict_list)):
            pred = predict_list[i].reshape(-1, 5+self.num_classes)
            anchor_box = self._generate_anchor_box(self.out_size[i], self.anchors[3*i:3*(i+1)]).reshape(-1, 4)

            pred_box = pred[..., :4]
            pred_conf = pred[..., 4]
            pred_cls = pred[..., 5:]

            pred_box[..., :2] += anchor_box[..., :2]
            # pred_box[..., 2:4] = torch.exp_(pred_box[..., 2:4]) * anchor_box[..., 2:4]

            conf_idx = pred_conf >= confidence_threshold
            pred_box = pred_box[conf_idx]
            pred_conf = pred_conf[conf_idx]
            pred_cls = pred_cls[conf_idx]

            pred_box[..., 0:3:2] /= self.out_size[i][0]
            pred_box[..., 1:4:2] /= self.out_size[i][1]

            box_list.append(pred_box)
            conf_list.append(pred_conf)
            cls_list.append(pred_cls)

        box = torch.cat(box_list, dim=0)
        conf = torch.cat(conf_list, dim=0)
        cls = torch.cat(cls_list, dim=0)
        if len(cls) > 0:
            cls = torch.argmax(cls, dim=-1)

        for i in range(self.num_classes):
            cls_idx = cls == i

            box = box[cls_idx]
            conf = conf[cls_idx]

            if len(box) > 0:
                box = convert_box_from_yxhw_to_xyxy(box)
                nms_idx = nms(box, conf, nms_threshold)
                box = box[nms_idx]
                conf = conf[nms_idx]

            box_per_cls_list.append(box)
            conf_per_cls_list.append(conf)

        return box_per_cls_list, conf_per_cls_list



if __name__ == '__main__':
    from torchsummary import summary
    model = YOLOV3Detector((8, 16), 3).cuda()
    x3 = torch.ones(2, 256, 32, 64).cuda()
    x4 = torch.ones(2, 512, 16, 32).cuda()
    x5 = torch.ones(2, 1024, 8, 16).cuda()
    model(x3, x4, x5)













