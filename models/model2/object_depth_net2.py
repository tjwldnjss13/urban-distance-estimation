import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pytorch_util import calculate_iou, convert_box_from_yxhw_to_yxyx
from utils.disparity import get_image_from_disparity
from utils.ssim import ssim

from models.model2.encoder import Encoder
from models.model2.detector import Detector
from models.model2.decoder import Decoder


class ObjectDepthNet2(nn.Module):
    def __init__(self, in_size, num_classes, mode=None):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mode = mode    # 'detect', 'depth', None(detect & depth)
        self.num_scale = 4

        self.encoder = Encoder()
        feat_size = tuple([x // 32 for x in in_size])
        self.detector = Detector(feat_size, num_classes)
        self.decoder = Decoder()

    def forward(self, x):
        en = self.encoder(x)

        if self.mode == 'depth':
            depth = self.decoder(*en)
            return depth
        else:
            detector = self.detector(en[-1])
            if self.mode == 'detect':
                return detector
            else:
                depth = self.decoder(*en)
                return detector, depth

    def loss_detector(self, predict, target):
        def modified_rmse_loss(predict, target, reduction='mean'):
            pred = torch.sqrt(predict)
            tar = torch.sqrt(target)

            return F.mse_loss(pred, tar, reduction=reduction)

        def cross_entropy_loss(predict, target, reduction='mean'):
            loss = -target * torch.log(predict + 1e-9)

            if reduction == 'sum':
                return loss.sum()
            elif reduction == 'mean':
                return loss.mean()
            else:
                return loss

        def binary_cross_entropy_loss(predict, target, reduction='mean'):
            loss = -(target * torch.log(predict + 1e-9) + (1 - target) * torch.log(1 - predict + 1e-9))

            if reduction == 'sum':
                return loss.sum()
            elif reduction == 'mean':
                return loss.mean()
            else:
                return loss

        lambda_box = 5
        lambda_obj = 5
        lambda_no_obj = 1
        lambda_cls = 1

        num_batch = predict.shape[0]

        pred_det = predict.reshape(num_batch, -1, 5 + self.detector.num_classes)
        tar_det = target.reshape(num_batch, -1, 5 + self.detector.num_classes)

        objs_mask = tar_det[..., 4] == 1
        no_objs_mask = tar_det[..., 4] == 0

        with torch.no_grad():
            ious_objs = calculate_iou(convert_box_from_yxhw_to_yxyx(pred_det[objs_mask][..., :4]),
                                      convert_box_from_yxhw_to_yxyx(tar_det[objs_mask][..., :4]))

        # loss_box = lambda_box * (F.mse_loss(pred_det[objs_mask][..., :2], tar_det[objs_mask][..., :2], reduction='sum') +
        #                          modified_rmse_loss(pred_det[objs_mask][..., 2:4], tar_det[objs_mask][..., 2:4], reduction='sum')) / num_batch
        # loss_obj = lambda_obj * cross_entropy_loss(pred_det[objs_mask][..., 4], ious_objs, reduction='sum') / num_batch
        # loss_no_obj = lambda_no_obj * binary_cross_entropy_loss(pred_det[no_objs_mask][..., 4], tar_det[no_objs_mask][..., 4], reduction='sum') / num_batch
        # loss_cls = lambda_cls * cross_entropy_loss(pred_det[objs_mask][..., 5:], tar_det[objs_mask][..., 5:], reduction='sum') / num_batch

        loss_box = lambda_box * (F.mse_loss(pred_det[objs_mask][..., :2], tar_det[objs_mask][..., :2], reduction='mean') +
                                 modified_rmse_loss(pred_det[objs_mask][..., 2:4], tar_det[objs_mask][..., 2:4], reduction='mean'))
        loss_obj = lambda_obj * cross_entropy_loss(pred_det[objs_mask][..., 4], ious_objs, reduction='mean') / num_batch
        loss_no_obj = lambda_no_obj * binary_cross_entropy_loss(pred_det[no_objs_mask][..., 4], tar_det[no_objs_mask][..., 4], reduction='mean')
        loss_cls = lambda_cls * cross_entropy_loss(pred_det[objs_mask][..., 5:], tar_det[objs_mask][..., 5:], reduction='mean')

        loss = loss_box + loss_obj + loss_no_obj + loss_cls

        return loss, loss_box.detach().cpu().item(), loss_obj.detach().cpu().item(), loss_no_obj.detach().cpu().item(), loss_cls.detach().cpu().item()

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

        imgl_list = get_image_pyramid(image_left, self.num_scale)
        imgr_list = get_image_pyramid(image_right, self.num_scale)

        pred_imgr_list = [get_image_from_disparity(imgl_list[i], dr_list[i]) for i in range(self.num_scale)]
        pred_imgl_list = [get_image_from_disparity(imgr_list[i], -dl_list[i]) for i in range(self.num_scale)]

        loss_ap = [min_appearance_matching_loss(imgr_list[i], pred_imgr_list[i]) + min_appearance_matching_loss(imgl_list[i], pred_imgl_list[i]) for i in range(self.num_scale)]
        loss_ds = [disparity_smoothness_loss(imgr_list[i], dr_list[i]) + disparity_smoothness_loss(imgl_list[i], dl_list[i]) for i in range(self.num_scale)]
        loss_lr = [left_right_disparity_consistency_loss(dr_list[i], dl_list[i]) for i in range(self.num_scale)]

        loss_ap = sum(loss_ap)
        loss_ds = sum(loss_ds)
        loss_lr = sum(loss_lr)

        loss_depth = loss_ap + loss_ds + loss_lr

        return loss_depth, loss_ap.detach().cpu().item(), loss_ds.detach().cpu().item(), loss_lr.detach().cpu().item()

    def loss(self, predict_detector, target_detector, image_left, image_right, disparities):
        loss_detector, _, _, _, _ = self.loss_detector(predict_detector, target_detector)
        loss_depth, _, _, _ = self.loss_depth(image_left, image_right, disparities)

        return loss_detector + loss_depth, loss_detector.detach().cpu().item(), loss_depth.detach().cpu().item()


if __name__ == '__main__':
    from torchsummary import summary
    detect_only = False
    model = ObjectDepthNet2((256, 512), 9).cuda()
    summary(model, (3, 256, 512))





























