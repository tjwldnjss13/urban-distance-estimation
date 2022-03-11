import torch

from dataset.kitti_detection_dataset import KITTIDetectionDataset
from utils.pytorch_util import calculate_iou_with_height_width_grid


class YOLOV3KITTIDataset(KITTIDetectionDataset):
    def __init__(
            self,
            root,
            names_path,
            for_train,
            img_size=(256, 512),
            augmentation=None,
            normalize=True,
            compact_class=False,
            stereo=True,
            class_idx=-1
    ):
        super().__init__(
            root=root,
            names_path=names_path,
            for_train=for_train,
            img_size=img_size,
            augmentation=augmentation,
            normalize=normalize,
            compact_class=compact_class,
            stereo=stereo,
            class_idx=class_idx
        )
        self.augmentation = augmentation

        # self.out_size = [(8, 16), (16, 32), (32, 64)]
        self.out_size = [(img_size[0] // 2 ** i, img_size[1] // 2 ** i) for i in range(5, 2, -1)]
        self.anchors = torch.Tensor([[0.22, 0.28], [0.48, 0.38], [0.78, 0.9],
                                     [0.15, 0.07], [0.11, 0.15], [0.29, 0.14],
                                     [0.03, 0.02], [0.07, 0.04], [0.06, 0.08]])

    def _generate_yolov3_target(self, bbox, label):
        num_target = len(bbox)
        num_scale = len(self.anchors) // 3
        num_anchor_box = len(self.anchors) // num_scale

        target = [torch.zeros((self.out_size[i][0], self.out_size[i][1], num_anchor_box, 5+self.num_classes)) for i in range(num_scale)]

        for tar_idx in range(num_target):
            l = label[tar_idx]
            b = bbox[tar_idx]
            b[..., 0:3:2] /= self.img_size[0]
            b[..., 1:4:2] /= self.img_size[1]
            cy = (b[..., 0] + b[..., 2]) / 2
            cx = (b[..., 1] + b[..., 3]) / 2
            h = b[2] - b[0]
            w = b[3] - b[1]
            hw = torch.cat([h.unsqueeze(0), w.unsqueeze(0)], dim=-1).unsqueeze(0)
            ious_box_anchor = calculate_iou_with_height_width_grid(hw, self.anchors)[0]
            idx_ious_box_anchor = torch.argsort(ious_box_anchor, descending=True)

            has_box = [False] * 3

            for idx in idx_ious_box_anchor:
                scale_idx = idx // 3
                anchor_idx = idx % 3

                scale_hw = self.out_size[scale_idx]
                y_idx = int(cy * scale_hw[0])
                x_idx = int(cx * scale_hw[1])
                y_val = cy * scale_hw[0] - y_idx
                x_val = cx * scale_hw[1] - x_idx
                h_val = torch.log(h / self.anchors[idx, 0] + 1e-16)
                w_val = torch.log(w / self.anchors[idx, 1] + 1e-16)
                # h_val = h * scale_hw[0]
                # w_val = w * scale_hw[1]

                coords = torch.tensor([y_val, x_val, h_val, w_val])

                if not has_box[scale_idx] and target[scale_idx][y_idx, x_idx, anchor_idx, 4] == 0:
                    target[scale_idx][y_idx, x_idx, anchor_idx, 0:4] = coords
                    target[scale_idx][y_idx, x_idx, anchor_idx, 4] = 1
                    target[scale_idx][y_idx, x_idx, anchor_idx, 5+l] = 1
                    has_box[scale_idx] = True
                    break

        for i in range(num_scale):
            h_scale, w_scale = self.out_size[i]
            target[i] = target[i].reshape(h_scale, w_scale, -1)

        return target

    def __getitem__(self, idx):
        ann = super().__getitem__(idx)
        img_left = ann['img_left']
        img_right = ann['img_right']
        bbox = ann['bbox']
        label = ann['label']

        target = self._generate_yolov3_target(bbox, label)

        ann_ret = {}
        ann_ret['imgl'] = img_left
        ann_ret['imgr'] = img_right

        num_scale = len(self.anchors) // 3
        for i in range(num_scale):
            name = 'target' + str(i+1)
            ann_ret[name] = target[i]

        return ann_ret


def custom_collate_fn(batch):
    return batch


if __name__ == '__main__':
    root = 'D://DeepLearningData/KITTI'
    name_pth = 'kitti_compact.name'
    dset = YOLOV3KITTIDataset(root=root, names_path=name_pth, compact_class=True)
    ann = dset[2]