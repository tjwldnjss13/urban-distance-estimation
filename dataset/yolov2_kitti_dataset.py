import torch

from dataset.kitti_detection_dataset import KITTIDetectionDataset
from utils.pytorch_util import calculate_iou


class YOLOKITTIDataset(KITTIDetectionDataset):
    def __init__(self, root, names_path, img_size=(256, 512), augmentation=None, normalize=True, compact_class=False):
        super().__init__(root=root, names_path=names_path, img_size=img_size, augmentation=augmentation, normalize=normalize, compact_class=compact_class)
        self.augmentation = augmentation

        self.out_size = (8, 16)
        self.anchors = torch.Tensor([[1.0655, 1.6272],
                                     [2.4673, 3.9295],
                                     [4.9840, 6.2226],
                                     [2.9788, 11.6568],
                                     [6.1582, 13.8294]])
        self.anchor_boxes = self._generate_anchor_box(self.out_size)

    def _generate_anchor_box(self, out_size):
        """
        Make anchor box the same shape as output's.
        :param anchor_box_sizes: tensor, [# anchor box, (height, width)]
        :param out_size: tuple or list, (height, width)

        :return tensor, [height, width, (cy, cx, h, w) * num bounding box]
        """

        out = torch.zeros(out_size[0], out_size[1], 4 * len(self.anchors)).cuda()
        cy_ones = torch.ones(out_size[0], 1)
        cx_ones = torch.ones(1, out_size[1])
        cy_tensor = torch.zeros(out_size[0], 1)
        cx_tensor = torch.zeros(1, out_size[1])

        for i in range(1, out_size[0]):
            cx_tensor = torch.cat([cx_tensor, cx_ones * i], dim=0)

        for i in range(1, out_size[1]):
            cy_tensor = torch.cat([cy_tensor, cy_ones * i], dim=1)

        ctr_tensor = torch.cat([cy_tensor.unsqueeze(2), cx_tensor.unsqueeze(2)], dim=2)

        for i in range(len(self.anchors)):
            out[:, :, 4 * i:4 * i + 2] = ctr_tensor
            out[:, :, 4 * i + 2] = self.anchors[i, 0]
            out[:, :, 4 * i + 3] = self.anchors[i, 1]

        return out

    def _generate_yolo_target(self, ground_truth_boxes, category_classes):
        """
        :param ground_truth_boxes: tensor, [num ground truth, (y1, x1, y2, x2)]
        :param category_classes: tensor, [num ground truth, class label]

        :return: tensor, [height of output, width of output, (cy, cx, h, w, p) * num bounding boxes]
        """

        gt_bboxes = ground_truth_boxes.to(self.device)
        n_bbox_predict = self.anchors.shape[0]

        n_gt = len(gt_bboxes)

        target = torch.zeros((*self.out_size, (5 + self.num_classes) * n_bbox_predict))

        box_assign = torch.zeros((*self.out_size, n_bbox_predict))

        for i in range(n_gt):
            gt = gt_bboxes[i]
            if len(gt) == 0:
                continue
            h_gt, w_gt = (gt[2] - gt[0]), (gt[3] - gt[1])
            y_gt, x_gt = (gt[0] + .5 * h_gt), (gt[1] + .5 * w_gt)

            y_idx, x_idx = int(y_gt), int(x_gt)

            if y_idx >= self.out_size[0] or x_idx >= self.out_size[1]:
                continue

            gt_anc_ious = calculate_iou(gt.unsqueeze(0), self.anchor_boxes[y_idx, x_idx].reshape(-1, 4), dim=1)
            _, gt_anc_ious_idx = torch.sort(gt_anc_ious, dim=-1, descending=True)

            for box_idx in gt_anc_ious_idx:
                if not box_assign[y_idx, x_idx, box_idx]:
                    box_assign[y_idx, x_idx, box_idx] = 1
                    anc_gt_idx = box_idx
                    break

            target[y_idx, x_idx, (5 + self.num_classes) * anc_gt_idx] = y_gt
            target[y_idx, x_idx, (5 + self.num_classes) * anc_gt_idx + 1] = x_gt
            target[y_idx, x_idx, (5 + self.num_classes) * anc_gt_idx + 2] = h_gt
            target[y_idx, x_idx, (5 + self.num_classes) * anc_gt_idx + 3] = w_gt
            target[y_idx, x_idx, (5 + self.num_classes) * anc_gt_idx + 4] = 1
            target[y_idx, x_idx, (5 + self.num_classes) * anc_gt_idx + 5 + category_classes[i]] = 1

        target = target.reshape(-1, 5+self.num_classes)

        return target

    def __getitem__(self, idx):
        ann = super().__getitem__(idx)
        img_left = ann['img_left']
        bbox = ann['bbox']
        label = ann['label']

        bbox[..., 0:3:2] *= self.out_size[0] / self.img_size[0]
        bbox[..., 1:4:2] *= self.out_size[1] / self.img_size[1]

        yolo_target = self._generate_yolo_target(bbox, label)

        ann_ret = {}
        ann_ret['img_left'] = img_left
        ann_ret['img_right'] = ann['img_right']
        ann_ret['detector_target'] = yolo_target
        ann_ret['bbox'] = bbox

        return ann_ret


def custom_collate_fn(batch):
    return batch


if __name__ == '__main__':
    root = 'D://DeepLearningData/KITTI'
    dset = YOLOKITTIDataset(root)
    ann = dset[0]