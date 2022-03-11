import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.colors as colors
import matplotlib.cm as cm

from utils.pytorch_util import calculate_iou
from utils.disparity import get_image_from_disparity
from utils.ssim import ssim
from metric import categorical_accuracy, confidence_accuracy

from models.model6.encoder_srdarknet import SRDarkNet
from models.model6.detector_yolov3 import YOLOV3Detector
from models.model6.decoder import Decoder


class UDNet6(nn.Module):
    def __init__(self, in_size, num_classes, mode=None):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mode = mode    # 'detect', 'depth', None(detect & depth)
        self.num_detector_scale = 3
        self.num_depth_scale = 4

        self.encoder = SRDarkNet()
        feat_size = tuple([i // 32 for i in in_size])
        self.detector = YOLOV3Detector(feat_size, num_classes)
        self.decoder = Decoder()

    def get_depth(self, disp):
        def visualize_depth(depth):
            vmax = np.percentile(depth, 95)
            norm = colors.Normalize(vmin=depth.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=norm, cmap='magma_r')
            depth = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)

            return depth

        dr, dl = disp.squeeze(0)
        dr = dr.detach().cpu().numpy()
        dl = dl.detach().cpu().numpy()

        dr = cv.resize(dr, (1242, 375))
        dl = cv.resize(dl, (1242, 375))

        dr = .54 * 721 / (1242 * dr)
        dl = .54 * 721 / (1242 * dl)

        # dr = visualize_depth(dr)
        # dl = visualize_depth(dl)

        return dr, dl

    def forward(self, x):
        feat = self.encoder(x)

        if self.mode == 'depth':
            depth = self.decoder(*feat)
            return depth
        else:
            detector = self.detector(*feat[3:])
            if self.mode == 'detect':
                return detector
            else:
                depth = self.decoder(*feat)
                return detector, depth

    def loss_detector(self, predict_list, target_list):
        def multi_l1_loss(predict, target, reduction='mean'):
            loss = sum([F.l1_loss(predict[..., i], target[..., i], reduction) for i in range(predict.shape[-1])])
            return loss

        def multi_root_l1_loss(predict, target, reduction='mean'):
            pred = torch.sqrt(predict)
            tar = torch.sqrt(target)

            loss = sum([F.l1_loss(pred[..., i], tar[..., i], reduction) for i in range(pred.shape[-1])])

            return loss

        def multi_bce_loss(predict, target, reduction='mean'):
            def bce(predict, target, reduction):
                ce = -(target * torch.log(predict + 1e-16) + (1 - target) * torch.log(1 - predict + 1e-16))

                if reduction == 'mean':
                    return ce.mean()
                if reduction == 'sum':
                    return ce.sum()
                return ce

            loss = sum([bce(predict[..., i], target[..., i], reduction) for i in range(predict.shape[-1])])

            return loss

        def sigmoid_inverse(x):
            return torch.log(x / (1 - x + 1e-16) + 1e-16)

        lambda_box = 1
        lambda_obj = 5
        lambda_no_obj = 1
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

        pred_box = torch.cat([
            pred[objs][..., :2],
            # torch.exp(pred[objs][..., 2:4]) * anchor[objs]
            torch.pow(torch.sigmoid(pred[objs][..., 2:4]) * 2, 3)
        ], dim=-1)

        tar_box = torch.cat([
            tar[objs][..., :2],
            # torch.exp(tar[objs][..., 2:4]) * anchor[objs]
            torch.pow(torch.sigmoid(tar[objs][..., 2:4]) * 2, 3)
        ], dim=-1)

        # loss_box = lambda_box * multi_l1_loss(pred_box[..., :2], tar_box[..., :2]) + multi_root_l1_loss(pred_box[..., 2:4], tar_box[..., 2:4])
        loss_box = lambda_box * multi_l1_loss(pred_box, tar_box)
        loss_obj = lambda_obj * F.l1_loss(pred[objs][..., 4], ious * tar[objs][..., 4])
        loss_no_obj = lambda_no_obj * F.binary_cross_entropy(pred[no_objs][..., 4], tar[no_objs][..., 4])
        loss_cls = lambda_cls * multi_bce_loss(pred[objs][..., 5:], tar[objs][..., 5:])

        # print()
        # print(loss_box.detach().cpu().item(), loss_obj.detach().cpu().item(), loss_no_obj.detach().cpu().item(), loss_cls.detach().cpu().item())

        loss = loss_box + loss_obj + loss_no_obj + loss_cls
        if torch.isnan(loss):
            if not self.training:
                print()
                for m in range(len(pred_box)):
                    print(pred_box[m].detach().cpu().numpy(), tar_box[m].detach().cpu().numpy())

        iou = torch.mean(ious)

        acc_conf = confidence_accuracy(pred[..., 4], tar[..., 4], .3)
        acc_cls = categorical_accuracy(pred[objs][..., 5:], tar[objs][..., 5:])

        return loss, loss_box.detach().cpu().item(), loss_obj.detach().cpu().item(), loss_no_obj.detach().cpu().item(), loss_cls.detach().cpu().item(), iou.detach().cpu().item(), acc_conf, acc_cls

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
        loss_detector, loss_box, loss_obj, loss_noobj, loss_cls, iou, acc_conf, acc_cls = self.loss_detector(predict_detector, target_detector)
        loss_depth = self.loss_depth(image_left, image_right, disparities)

        loss = loss_detector + loss_depth

        return loss, loss_detector.detach().cpu().item(), loss_box, loss_obj, loss_noobj, loss_cls, loss_depth.detach().cpu().item(), iou, acc_conf, acc_cls


if __name__ == '__main__':
    from torchsummary import summary
    model = UDNet6((256, 512), 2).cuda()
    x = torch.ones(2, 3, 256, 512).cuda()
    det, disp = model(x)

