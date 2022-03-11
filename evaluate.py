import argparse
import torch
import cv2 as cv
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score

from models.model9.udnet9 import UDNet9
from dataset.kitti_detection_dataset import KITTIDetectionDataset
from utils.pytorch_util import calculate_ious_grid, convert_box_from_yxyx_to_xyxy
from metric import mean_average_precision


def get_kitti_dataset(args):
    dset = KITTIDetectionDataset(
        root=args.dataset_dir,
        names_path=args.name_pth,
        for_train=False,
        img_size=args.in_size,
        compact_class=True,
        normalize=False,
        stereo=False,
        class_idx=-1
    )

    return dset


def get_average_precision(
        pred_box_list,
        tar_box_list,
        iou_thresh
):
    ious_list = []
    tp_list = []
    fp_list = []
    num_tar = 0

    for i in range(len(pred_box_list)):
        print(f'{i + 1}/{len(pred_box_list)}', end='\r')

        pred_boxes = pred_box_list[i]
        tar_boxes = tar_box_list[i]

        tar_covered = torch.zeros(len(tar_boxes))
        tp = torch.zeros(len(pred_boxes))
        fp = torch.zeros(len(pred_boxes))

        if len(tar_boxes) > 0 and len(pred_boxes) > 0:
            ious_pred_tar = calculate_ious_grid(pred_boxes, tar_boxes, box_format='xyxy')
            ious_pred_tar, idx_pred_tar = torch.sort(ious_pred_tar, dim=-1, descending=True)
            ious_pred_tar = ious_pred_tar[..., 0]
            idx_pred_tar = idx_pred_tar[..., 0]
            # ious_pred_tar, idx_ord = torch.sort(ious_pred_tar, dim=0, descending=True)
            # idx_pred_tar = idx_pred_tar[idx_ord]

            ious_list.append(ious_pred_tar)

            for j, iou in enumerate(ious_pred_tar):
                if iou > iou_thresh:
                    if tar_covered[idx_pred_tar[j]] == 0:
                        tp[j] = 1
                        tar_covered[idx_pred_tar[j]] = 1
                    else:
                        fp[j] = 1
                else:
                    fp[j] = 1
        else:
            fp = torch.ones(len(pred_boxes))

        tp_list.append(tp)
        fp_list.append(fp)
        num_tar += len(tar_boxes)

        ious = torch.cat(ious_list)
        tp = torch.cat(tp_list)
        fp = torch.cat(fp_list)

        ious, idx_ious = torch.sort(ious, descending=True)
        tp = tp[idx_ious]
        fp = fp[idx_ious]

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        precisions = torch.divide(tp_cumsum, (tp_cumsum + fp_cumsum + 1e-12))
        precisions = torch.cat([torch.tensor([1]), precisions])
        recalls = tp_cumsum / (num_tar + 1e-12)
        recalls = torch.cat([torch.tensor([0]), recalls])
        average_precision = torch.trapz(precisions, recalls)

        plt.plot(recalls.numpy(), precisions.numpy(), 'r-')
        plt.savefig(args.result_dir + f'{average_precision:.2f}ap_{args.iou_thresh}iou_thresh.png')

        print('precisions :', precisions)
        print('recalls :', recalls)
        print('average precision :', average_precision)


def get_mean_average_precision(
        pred_box_list,
        tar_box_list,
        pred_label_list,
        tar_label_list,
        iou_thresh,
        num_classes
):
    ious_list = [[] for _ in range(num_classes)]
    tp_list = [[] for _ in range(num_classes)]
    fp_list = [[] for _ in range(num_classes)]
    num_tar = [0 for _ in range(num_classes)]

    for c in range(num_classes):
        for i in range(len(pred_box_list)):
            print(f'{i+1}/{len(pred_box_list)}', end='\r')

            pred_boxes = pred_box_list[i]
            tar_boxes = tar_box_list[i]
            pred_label = pred_label_list[i]
            tar_label = tar_label_list[i]

            pred_idx = pred_label == c
            tar_idx = tar_label == c

            if len(pred_boxes) == 0:
                print(pred_boxes.shape)
                print(pred_boxes)
                print(pred_label.shape)
                print(pred_label)
                print(pred_idx.shape)
                print(pred_idx)
                print(pred_boxes[pred_idx].shape)
                print(pred_boxes[pred_idx])

            tar_covered = torch.zeros(len(tar_boxes[tar_idx]))
            tp = torch.zeros(len(pred_boxes[pred_idx]))
            fp = torch.zeros(len(pred_boxes[pred_idx]))

            if len(tar_boxes[tar_idx]) > 0 and len(pred_boxes[pred_idx]) > 0:
                ious_pred_tar = calculate_ious_grid(pred_boxes[pred_idx], tar_boxes[tar_idx], box_format='xyxy')
                ious_pred_tar, idx_pred_tar = torch.sort(ious_pred_tar, dim=-1, descending=True)
                ious_pred_tar = ious_pred_tar[..., 0]
                idx_pred_tar = idx_pred_tar[..., 0]
                # ious_pred_tar, idx_ord = torch.sort(ious_pred_tar, dim=0, descending=True)
                # idx_pred_tar = idx_pred_tar[idx_ord]

                ious_list[c].append(ious_pred_tar)

                for j, iou in enumerate(ious_pred_tar):
                    if iou > iou_thresh:
                        if tar_covered[idx_pred_tar[j]] == 0:
                            tp[j] = 1
                            tar_covered[idx_pred_tar[j]] = 1
                        else:
                            fp[j] = 1
                    else:
                        fp[j] = 1
            else:
                fp = torch.ones(len(pred_boxes[pred_idx]))

            tp_list[c].append(tp)
            fp_list[c].append(fp)
            num_tar[c] += len(tar_boxes[tar_idx])

    average_precision_list = []

    for c in range(num_classes):
        ious = torch.cat(ious_list[c])
        tp = torch.cat(tp_list[c])
        fp = torch.cat(fp_list[c])

        ious, idx_ious = torch.sort(ious, descending=True)
        tp = tp[idx_ious]
        fp = fp[idx_ious]

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        precisions = torch.divide(tp_cumsum, (tp_cumsum + fp_cumsum + 1e-12))
        precisions = torch.cat([torch.tensor([1]), precisions])
        recalls = tp_cumsum / (num_tar[c] + 1e-12)
        recalls = torch.cat([torch.tensor([0]), recalls])
        average_precision = torch.trapz(precisions, recalls)

        average_precision_list.append(average_precision)

        plt.plot(recalls.numpy(), precisions.numpy(), 'r-')
        plt.savefig(args.result_dir+f'{c}class_{average_precision:.2f}map_{args.iou_thresh}iou_thresh.png')

        print('precisions :', precisions)
        print('recalls :', recalls)
        print('average precision :', average_precision)

    print('mAP :', sum(average_precision_list) / len(average_precision_list))


def eval_detection(args, model):
    dset = get_kitti_dataset(args)

    pred_box_list = []
    tar_box_list = []
    pred_label_list = []
    tar_label_list = []

    for i in range(len(dset)):
        print(f'{i+1}/{len(dset)}', end='\r')

        ann = dset[i]
        img = ann['img_left'].unsqueeze(0).to(args.device)
        tar_box = convert_box_from_yxyx_to_xyxy(ann['bbox'])
        tar_label = ann['label']

        detector, depth = model(img)
        pred_box, pred_conf, pred_cls = model.detector.get_output(detector, args.conf_thresh, args.nms_thresh)

        pred_box_list.append(pred_box.detach().cpu())
        tar_box_list.append(tar_box)

        pred_label_list.append(pred_cls.detach().cpu())
        tar_label_list.append(tar_label)

    mean_average_precision(
        pred_box_list,
        tar_box_list,
        pred_label_list,
        tar_label_list,
        args.iou_thresh,
        args.num_classes
    )

    exit()


def main(args):
    model = UDNet9(
        in_size=args.in_size,
        num_classes=args.num_classes
    ).to(args.device)
    model.load_state_dict(torch.load(args.weight_pth, map_location=args.device)['model_state_dict'])
    model.eval()

    eval_detection(args, model)


if __name__ == '__main__':
    def type_device(device):
        if device == 'cuda:0' or device == 'cpu':
            return torch.device(device)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', required=False, type=type_device, default=device)
    parser.add_argument('--weight_pth', required=False, type=str, default='./weights/(2nd)UDNet9_YOLOV3KITTIDataset_31epoch_0.000010_1.224loss(train)_1.661loss.ckpt')
    parser.add_argument('--in_size', required=False, type=tuple, default=(256, 512))
    parser.add_argument('--num_classes', required=False, type=int, default=2)
    parser.add_argument('--conf_thresh', required=False, type=float, default=.86)
    parser.add_argument('--nms_thresh', required=False, type=float, default=.3)
    parser.add_argument('--iou_thresh', required=False, type=float, default=.55)
    parser.add_argument('--dataset_dir', required=False, type=str, default='D://DeepLearningData/KITTI/')
    parser.add_argument('--name_pth', required=False, type=str, default='./dataset/kitti_compact.name')
    parser.add_argument('--result_dir', required=False, type=str, default='./result/')

    args = parser.parse_args()

    main(args)