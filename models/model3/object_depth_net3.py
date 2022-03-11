import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pytorch_util import calculate_iou
from utils.disparity import get_image_from_disparity
from utils.ssim import ssim

from models.model3.encoder_darknet import MyDarknet
from models.model3.encoder_densenet import DenseNet
from models.model3.detector_yolov3 import YOLOV3Detector
from models.model3.decoder import Decoder
from utils.pytorch_util import calculate_iou


class ObjectDepthNet3(nn.Module):
    def __init__(self, in_size, num_classes, mode=None):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mode = mode    # 'detect', 'depth', None(detect & depth)
        self.num_detector_scale = 3
        self.num_depth_scale = 4

        self.encoder = DenseNet(32, .5)
        feat_size = tuple([x // 32 for x in in_size])
        self.detector = YOLOV3Detector(feat_size, num_classes)
        self.decoder = Decoder()

    def forward(self, x):
        feat_detector, feat_depth = self.encoder(x)

        if self.mode == 'depth':
            depth = self.decoder(*feat_depth)
            return depth
        else:
            detector = self.detector(*feat_detector)
            if self.mode == 'detect':
                return detector
            else:
                depth = self.decoder(*feat_depth)
                return detector, depth

    def loss_detector(self, predict_list, target_list):
        def mean_square_root_error_loss(predict, target, reduction='mean'):
            pred = torch.sqrt(predict)
            tar = torch.sqrt(target)

            return F.mse_loss(pred, tar, reduction=reduction)

        def root_l1_loss(predict, target, reduction='mean'):
            pred = torch.sqrt(predict)
            tar = torch.sqrt(target)

            return F.l1_loss(pred, tar, reduction=reduction)

        def cross_entropy_loss(predict, target, reduction='mean'):
            loss = -target * torch.log(predict + 1e-16)

            if reduction == 'sum':
                return loss.sum()
            elif reduction == 'mean':
                return loss.mean()
            else:
                return loss

        def binary_cross_entropy_loss(predict, target, reduction='mean'):
            loss = -(target * torch.log(predict + 1e-16) + (1 - target) * torch.log(1 - predict + 1e-16))

            if reduction == 'sum':
                return loss.sum()
            elif reduction == 'mean':
                return loss.mean()
            else:
                return loss

        def sigmoid_inverse(x):
            return torch.log(x / (1 - x + 1e-16) + 1e-16)

        lambda_box = 1
        lambda_obj = 5
        lambda_no_obj = .0001
        lambda_cls = 1

        pred_list = []
        tar_list = []
        anchor_list = []

        for i in range(len(predict_list)):
            pred = predict_list[i]
            tar = target_list[i]

            B, H, W = pred.shape[:3]

            pred = pred.reshape(-1, 5+self.detector.num_classes)
            tar = tar.reshape(-1, 5+self.detector.num_classes)

            anchor = self.detector.anchors[3*i:3*(i+1)].reshape(-1).unsqueeze(0).repeat(B, H, W, 1).reshape(-1, 2)
            anchor[..., 0] *= H
            anchor[..., 1] *= W

            pred_list.append(pred)
            tar_list.append(tar)
            anchor_list.append(anchor)

        pred = torch.cat(pred_list, dim=0)
        tar = torch.cat(tar_list, dim=0)
        anchor = torch.cat(anchor_list, dim=0)

        objs = tar[..., 4] == 1
        no_objs = tar[..., 4] == 0

        with torch.no_grad():
            pred_box = torch.cat([
                pred[objs][..., :2],
                torch.exp(pred[objs][..., 2:4]) * anchor[objs]
            ], dim=-1)
            tar_box = torch.cat([
                tar[objs][..., :2],
                torch.exp(tar[objs][..., 2:4]) * anchor[objs]
            ], dim=-1)
            ious = calculate_iou(pred_box, tar_box, box_format='yxhw')

        # pred_box = torch.cat([
        #     pred[objs][..., :2],
        #     torch.pow(torch.sigmoid(pred[objs][..., 2:4]) * 2, 3)
        # ], dim=-1)
        pred_box = torch.cat([
            pred[objs][..., :2],
            torch.exp(pred[objs][..., 2:4]) * anchor[objs]
        ], dim=-1)
        # pred_box = pred[objs][..., :4]

        # tar_box = torch.cat([
        #     tar[objs][..., :2],
        #     torch.pow(torch.sigmoid(tar[objs][..., 2:4]) * 2, 3)
        # ], dim=-1)
        # tar_box = torch.cat([
        #     sigmoid_inverse(tar[objs][..., :2]),
        #     tar[objs][..., 2:4]
        # ], dim=-1)
        tar_box = torch.cat([
            tar[objs][..., :2],
            torch.exp(tar[objs][..., 2:4]) * anchor[objs]
        ], dim=-1)
        # tar_box = tar[objs][..., :4]

        # loss_box = lambda_box * F.l1_loss(pred_box, tar_box, reduction='sum') / B
        # loss_obj = lambda_obj * cross_entropy_loss(pred[objs][..., 4], ious, reduction='sum') / B
        # loss_no_obj = lambda_no_obj * F.binary_cross_entropy(pred[no_objs][..., 4], tar[no_objs][..., 4], reduction='sum') / B
        # loss_cls = lambda_cls * F.binary_cross_entropy(pred[objs][..., 5:], tar[objs][..., 5:], reduction='sum') / B

        # loss_box = F.l1_loss(pred_box, tar_box, reduction='mean')
        loss_box = (F.l1_loss(pred_box[..., :2], tar_box[..., :2], reduction='mean') + root_l1_loss(pred_box[..., 2:4], tar_box[..., 2:4], reduction='mean')) / 2
        loss_obj = F.l1_loss(pred[objs][..., 4], ious, reduction='mean')
        loss_no_obj = F.binary_cross_entropy(pred[no_objs][..., 4], tar[no_objs][..., 4], reduction='mean')
        loss_cls = F.binary_cross_entropy(pred[objs][..., 5:], tar[objs][..., 5:], reduction='mean')

        # print()
        # print(loss_box.detach().cpu().item(), loss_obj.detach().cpu().item(), loss_no_obj.detach().cpu().item(), loss_cls.detach().cpu().item())

        loss = loss_box + loss_obj + loss_no_obj + loss_cls
        if torch.isnan(loss):
            if not self.training:
                print()
                for m in range(len(pred_box)):
                    print(pred_box[m].detach().cpu().numpy(), tar_box[m].detach().cpu().numpy())

        iou = torch.mean(ious)

        if self.mode is None:
            return loss, iou.detach().cpu().item()

        return loss, loss_box.detach().cpu().item(), loss_obj.detach().cpu().item(), loss_no_obj.detach().cpu().item(), loss_cls.detach().cpu().item(), iou.detach().cpu().item()

    def loss_depth(self, image_left, image_right, disparities):
        def get_image_derivative_x(image, filter=None):
            """
            :param image: tensor, [num batches, channels, height, width]
            :param filter: tensor, [num_batches=1, channels, height, width]
            :return:
            """
            if filter is None:
                filter = torch.Tensor([[[[-1, 0, 1],
                                         [-2, 0, 2],
                                         [-1, 0, 1]]]]).to(image.device)

            num_channels = image.shape[1]
            if num_channels > 1:
                filter = torch.cat([filter for _ in range(num_channels)], dim=1)

            derv_x = F.conv2d(image, filter, None, 1, 1)

            return derv_x

        def get_image_derivative_y(image, filter=None):
            """
            :param image: tensor, [num batches, channels, height, width]
            :param filter: tensor, [num_batches=1, channels, height, width]
            :return:
            """
            if filter is None:
                filter = torch.Tensor([[[[-1, -2, -1],
                                         [0, 0, 0],
                                         [1, 2, 1]]]]).to(image.device)

            num_channels = image.shape[1]
            if num_channels > 1:
                filter = torch.cat([filter for _ in range(num_channels)], dim=1)

            derv_y = F.conv2d(image, filter, None, 1, 1)

            return derv_y

        def min_appearance_matching_loss(image1, image2, alpha=.85):
            """
            :param image1: tensor, [num batches, channels, height, width]
            :param image2: tensor, [num_batches, channels, height, width]
            :param alpha: float, 0~1
            :return:
            """
            assert image1.shape == image2.shape

            N_batch, _, h, w = image1.shape
            N_pixel = h * w

            loss_ssim = alpha * ((1 - ssim(image1, image2, 3)) / 2).min()
            loss_l1 = (1 - alpha) * torch.abs(image1 - image2).min()
            loss = loss_ssim + loss_l1

            # print(f' ssim: {ssim(image1, image2, 3).detach().cpu().numpy()} loss_sim: {loss_ssim.detach().cpu().numpy()} \
            #       loss_l1: {loss_l1.detach().cpu().numpy()} loss: {loss.detach().cpu().numpy()}')

            return loss

        def disparity_smoothness_loss(image, disparity_map):
            """
            :param image: tensor, [num batches, channels, height, width]
            :param disparity_map: tensor, [num batches, channels, height, width]
            :return:
            """
            img = image
            dmap = disparity_map

            N_batch = image.shape[0]
            N_pixel = image.shape[2] * image.shape[3]

            grad_dmap_x = get_image_derivative_x(dmap)
            grad_dmap_y = get_image_derivative_y(dmap)

            grad_img_x = get_image_derivative_x(img)
            grad_img_y = get_image_derivative_y(img)

            grad_img_x = torch.abs(grad_img_x).sum(dim=1).unsqueeze(1)
            grad_img_y = torch.abs(grad_img_y).sum(dim=1).unsqueeze(1)

            loss = (torch.abs(grad_dmap_x) * torch.exp(-torch.abs(grad_img_x)) +
                    torch.abs(grad_dmap_y) * torch.exp(-torch.abs(grad_img_y))).mean()

            return loss

        def left_right_disparity_consistency_loss(disparity_map_left, disparity_map_right):
            assert disparity_map_left.shape == disparity_map_right.shape

            dl = disparity_map_left
            dr = disparity_map_right

            dl_cons = get_image_from_disparity(dr, -dl)
            dr_cons = get_image_from_disparity(dl, dr)

            loss_l = torch.mean(torch.abs(dl_cons - dl))
            loss_r = torch.mean(torch.abs(dr_cons - dr))

            loss = (loss_l + loss_r).sum()

            return loss

        def get_image_pyramid(image, num_scale):
            images_pyramid = []
            h, w = image.shape[2:]
            for i in range(num_scale):
                h_scale, w_scale = h // (2 ** i), w // (2 ** i)
                images_pyramid.append(F.interpolate(image, size=(h_scale, w_scale), mode='bilinear', align_corners=True))

            return images_pyramid

        dr_list = [d[:, 0].unsqueeze(1) for d in disparities]
        dl_list = [d[:, 1].unsqueeze(1) for d in disparities]

        imgl_list = get_image_pyramid(image_left, self.num_depth_scale)
        imgr_list = get_image_pyramid(image_right, self.num_depth_scale)

        pred_imgr_list = [get_image_from_disparity(imgl_list[i], dr_list[i]) for i in range(self.num_depth_scale)]
        pred_imgl_list = [get_image_from_disparity(imgr_list[i], -dl_list[i]) for i in range(self.num_depth_scale)]

        loss_ap = [min_appearance_matching_loss(imgr_list[i], pred_imgr_list[i]) + min_appearance_matching_loss(imgl_list[i], pred_imgl_list[i]) for i in range(self.num_depth_scale)]
        loss_ds = [disparity_smoothness_loss(imgr_list[i], dr_list[i]) + disparity_smoothness_loss(imgl_list[i], dl_list[i]) for i in range(self.num_depth_scale)]
        loss_lr = [left_right_disparity_consistency_loss(dr_list[i], dl_list[i]) for i in range(self.num_depth_scale)]

        loss_ap = sum(loss_ap)
        loss_ds = sum(loss_ds)
        loss_lr = sum(loss_lr)

        loss_depth = loss_ap + loss_ds + loss_lr

        if self.mode is None:
            return loss_depth

        return loss_depth, loss_ap.detach().cpu().item(), loss_ds.detach().cpu().item(), loss_lr.detach().cpu().item()

    def loss(self, predict_detector, target_detector, image_left, image_right, disparities):
        loss_detector, iou = self.loss_detector(predict_detector, target_detector)
        loss_depth = self.loss_depth(image_left, image_right, disparities)

        loss = loss_detector + loss_depth

        return loss, loss_detector.detach().cpu().item(), loss_depth.detach().cpu().item(), iou


if __name__ == '__main__':
    from torchsummary import summary
    model = ObjectDepthNet3((256, 512), 3).cuda()
    x = torch.ones(2, 3, 256, 512).cuda()
    det, disp = model(x)

