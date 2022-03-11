import os
import cv2 as cv
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T

from dataset.augment import stereo_random_hsv

# For test
import matplotlib.pyplot as plt


class KITTIDetectionDataset(data.Dataset):
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
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.root = root
        self.for_train = for_train
        self.img_size = img_size
        self.augmentation = augmentation
        self.compact_class = compact_class
        self.stereo = stereo
        self.class_idx = class_idx

        with open(names_path, 'r') as f:
            self.label_names = np.array(list(map(str.strip, f.readlines())))
        self.num_classes = len(self.label_names)
        self.anns = self._load_annotations()

        num_train = int(len(self.anns) * .9)
        if for_train:
            self.anns = self.anns[:num_train]
        else:
            self.anns = self.anns[num_train:]

        if normalize:
            self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transform = T.ToTensor()

    def _convert_to_compact_class(self, class_label_str):
        cvt_cls_name_dict = {'Pedestrian': 'Person',
                             'Person_sitting': 'Person',
                             'Cyclist': 'Person',
                             'Car': 'Vehicle',
                             'Truck': 'Vehicle',
                             'Van': 'Vehicle',
                             'Tram': 'Vehicle',
                             'Misc': 'Vehicle',
                             'DontCare': 'DontCare'}
        return cvt_cls_name_dict[class_label_str]

    def _load_annotations(self):
        print('Loading annotations...')

        ann_list = []
        ann_dir = os.path.join(self.root, 'detection', 'data_object_label_2', 'training', 'label_2')
        img_left_dir = os.path.join(self.root, 'detection', 'data_object_image_2', 'training', 'image_2')
        if self.stereo:
            img_right_dir = os.path.join(self.root, 'detection', 'data_object_image_3', 'training', 'image_3')
        ann_fn_list = os.listdir(ann_dir)
        for ann_fn in ann_fn_list:
            with open(os.path.join(ann_dir, ann_fn), 'r') as f:
                lines = f.readlines()

            label_names = []
            labels = []
            bboxes = []
            for line in lines:
                infos = line.strip().split()
                label_name, x1, y1, x2, y2 = infos[0], *list(map(float, infos[4:8]))
                if label_name == 'DontCare':
                    continue

                if self.compact_class:
                    label_name = self._convert_to_compact_class(label_name)

                label = np.where(self.label_names == label_name)[0][0]

                label_names.append(label_name)
                labels.append(label)
                bboxes.append([y1, x1, y2, x2])

            img_fn = ann_fn.strip().split('.')[0] + '.png'
            img_left_pth = os.path.join(img_left_dir, img_fn)
            if self.stereo:
                img_right_pth = os.path.join(img_right_dir, img_fn)

            if self.class_idx >= 0:
                if self.class_idx not in labels:
                    continue

            ann = {}
            ann['img_left_pth'] = img_left_pth
            ann['label_name'] = label_names
            ann['label'] = labels
            ann['bbox'] = bboxes
            if self.stereo:
                ann['img_right_pth'] = img_right_pth

            ann_list.append(ann)

        print('Annotations loaded.')

        return ann_list

    def __getitem__(self, idx):
        ann = self.anns[idx]

        img_left_pth = ann['img_left_pth']
        img_left = cv.imread(img_left_pth)
        img_left = cv.cvtColor(img_left, cv.COLOR_BGR2RGB)

        if self.stereo:
            img_right_pth = ann['img_right_pth']
            img_right = cv.imread(img_right_pth)
            img_right = cv.cvtColor(img_right, cv.COLOR_BGR2RGB)

        h_og, w_og = img_left.shape[:2]
        h, w = self.img_size

        img_left = cv.resize(img_left, (w, h))
        if self.stereo:
            img_right = cv.resize(img_right, (w, h))

        bboxes = np.array(ann['bbox'])
        for i, b in enumerate(bboxes):
            b[0:3:2] = b[0:3:2] * h / h_og
            b[1:4:2] = b[1:4:2] * w / w_og
            bboxes[i] = b

        if self.for_train and self.stereo:
            img_left, img_right = stereo_random_hsv(img_left, img_right)
            if np.random.random() < .5:
                img_left, img_right = cv.flip(img_right, 1), cv.flip(img_left, 1)

        img_left = self.transform(img_left)
        if self.stereo:
            img_right = self.transform(img_right)
        bboxes = torch.as_tensor(bboxes)

        if self.augmentation:
            for aug in self.augmentation:
                img_left, bboxes = aug(img_left, bboxes)

        ann_ret = {}
        ann_ret['img_left'] = img_left
        ann_ret['bbox'] = bboxes
        ann_ret['label'] = torch.as_tensor(ann['label'])
        if self.stereo:
            ann_ret['img_right'] = img_right

        return ann_ret

    def __len__(self):
        return len(self.anns)

    def show_class_distribution(self):
        class_distribution = [0 for _ in range(self.num_classes)]

        for ann in self.anns:
            for label in ann['label']:
                class_distribution[label] += 1

        for i in range(self.num_classes):
            print(f'{self.label_names[i]} : {class_distribution[i]}')

    def display_box(self, idx):
        ann = self.anns[idx]

        img = cv.imread(ann['img_left_pth'])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        bbox = ann['bbox']
        for i in range(len(bbox)):
            y1, x1, y2, x2 = list(map(int, bbox[i]))
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    from torch.utils.data import Subset
    root = 'D://DeepLearningData/KITTI/'
    cls_name_pth = 'kitti_compact.name'

    # dset = KITTIDetectionDataset(root, cls_name_pth, compact_class=True, for_train=False, class_idx=-1)
    # print(len(dset))
    # dset.show_class_distribution()
    # print('---------------------------')
    dset = KITTIDetectionDataset(root, cls_name_pth, compact_class=True, for_train=True, class_idx=0)
    print(len(dset))
    dset.show_class_distribution()
    # print('---------------------------')
    # dset = KITTIDetectionDataset(root, cls_name_pth, compact_class=True, for_train=False, class_idx=1)
    # print(len(dset))
    # dset.show_class_distribution()
