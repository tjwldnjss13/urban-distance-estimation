import torch


from utils.pytorch_util import calculate_ious_grid


def categorical_accuracy(predict, target):
    pred = torch.argmax(predict, dim=-1)
    tar = torch.argmax(target, dim=-1)

    correct = pred == tar
    correct = correct.reshape(-1)

    acc = correct.sum() / len(correct)

    return acc.detach().cpu().item()


def confidence_accuracy(predict, target, threshold):
    pred = predict >= threshold
    tar = target.bool()

    correct = pred == tar
    correct = correct.reshape(-1)

    acc = correct.sum() / len(correct)

    return acc.detach().cpu().item()


def mean_average_precision(
        pred_box_list,
        tar_box_list,
        pred_label_list,
        tar_label_list,
        iou_thresh,
        num_classes
):
    ious_list = []
    tp_list = []
    fp_list = []
    num_tar = 0

    for i in range(len(pred_box_list)):
        print(f'{i+1}/{len(pred_box_list)}', end='\r')

        pred_boxes = pred_box_list[i]
        tar_boxes = tar_box_list[i]
        pred_label = pred_label_list[i]
        tar_label = tar_label_list[i]

        tar_covered = torch.zeros(len(tar_boxes))
        tp = torch.zeros(len(pred_boxes))
        fp = torch.zeros(len(pred_boxes))

        if len(tar_boxes) > 0 and len(pred_boxes) > 0:
            ious_pred_tar = calculate_ious_grid(pred_boxes, tar_boxes, box_format='xyxy')
            ious_pred_tar, idx_pred_tar = torch.sort(ious_pred_tar, dim=-1, descending=True)
            ious_pred_tar = ious_pred_tar[..., 0]
            idx_pred_tar = idx_pred_tar[..., 0]

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

    print(tp_list)
    print(fp_list)

    ious = torch.cat(ious_list)
    tp = torch.cat(tp_list)
    fp = torch.cat(fp_list)

    print(ious)
    ious, idx_ious = torch.sort(ious, descending=True)
    tp = tp[idx_ious]
    fp = fp[idx_ious]
    print(ious)
    print(idx_ious)
    print(tp)
    print(fp)

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    print(tp_cumsum)
    print(fp_cumsum)

    precisions = torch.divide(tp_cumsum, (tp_cumsum + fp_cumsum + 1e-12))
    precisions = torch.cat([torch.tensor([1]), precisions])
    recalls = tp_cumsum / (num_tar + 1e-12)
    recalls = torch.cat([torch.tensor([0]), recalls])
    ap = torch.trapz(precisions, recalls)

    print(precisions)
    print(recalls)
    print(ap)

    return ap


def mean_average_precision(
        pred_box_list,
        tar_box_list,
        pred_label_list,
        tar_label_list,
        iou_thresh,
        num_classes
):
    ious_list = []
    tp_list = []
    fp_list = []
    num_tar = 0

    for i in range(len(pred_box_list)):
        print(f'{i+1}/{len(pred_box_list)}', end='\r')

        pred_boxes = pred_box_list[i]
        tar_boxes = tar_box_list[i]
        pred_label = pred_label_list[i]
        tar_label = tar_label_list[i]

        for c in range(num_classes):
            pred_idx = pred_label == c
            tar_idx = tar_label == c

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
                fp = torch.ones(len(pred_boxes[pred_idx]))

            tp_list.append(tp)
            fp_list.append(fp)
            num_tar += len(tar_boxes[tar_idx])

    # print(tp_list)
    # print(fp_list)

    ious = torch.cat(ious_list)
    tp = torch.cat(tp_list)
    fp = torch.cat(fp_list)

    # print(ious)
    ious, idx_ious = torch.sort(ious, descending=True)
    tp = tp[idx_ious]
    fp = fp[idx_ious]
    # print(ious)
    # print(idx_ious)
    # print(tp)
    # print(fp)

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    # print(tp_cumsum)
    # print(fp_cumsum)

    precisions = torch.divide(tp_cumsum, (tp_cumsum + fp_cumsum + 1e-12))
    precisions = torch.cat([torch.tensor([1]), precisions])
    recalls = tp_cumsum / (num_tar + 1e-12)
    recalls = torch.cat([torch.tensor([0]), recalls])
    ap = torch.trapz(precisions, recalls)

    # print(precisions)
    # print(recalls)
    # print(ap)

    return ap


def mean_average_precision(
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
    num_tar_list = [0 for _ in range(num_classes)]

    for i in range(len(pred_box_list)):
        print(f'{i+1}/{len(pred_box_list)}', end='\r')

        pred_boxes = pred_box_list[i]
        tar_boxes = tar_box_list[i]
        pred_label = pred_label_list[i]
        tar_label = tar_label_list[i]

        for c in range(num_classes):
            pred_idx = pred_label == c
            tar_idx = tar_label == c

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
            num_tar_list[c] += len(tar_boxes[tar_idx])

    ap_list = []

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
        recalls = tp_cumsum / (num_tar_list[c] + 1e-12)
        recalls = torch.cat([torch.tensor([0]), recalls])
        ap = torch.trapz(precisions, recalls)
        ap_list.append(ap)

        print('tp_cumsum :', tp_cumsum.type(torch.int).numpy(), len(tp_cumsum))
        print('fp_cumsum :', fp_cumsum.type(torch.int).numpy(), len(fp_cumsum))
        print('precisions :', precisions.numpy(), len(precisions))
        print('recalls :', recalls.numpy(), len(recalls))
        print('num_tar :', num_tar_list[c])
        print('ap :', ap.item())
        print()

    map = sum(ap_list) / len(ap_list)
    print('mAP :', map.item())

    return map


if __name__ == '__main__':
    pred = torch.Tensor([.4, .6, .1, .7])
    tar = torch.Tensor([0, 1, 0, 1])

    acc = confidence_accuracy(pred, tar, .3)
    print(acc)