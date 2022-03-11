import argparse
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image
from torchvision.ops import nms

from dataset.kitti_detection_dataset import KITTIDetectionDataset
from utils.pytorch_util import convert_box_from_yxhw_to_xyxy

# from models.model3.object_depth_net3 import ObjectDepthNet3
from models.model6.udnet6 import UDNet6
from models.model7.udnet7 import UDNet7
from models.model8.udnet8 import UDNet8
from models.model9.udnet9 import UDNet9


def get_dataset(args):
    dset = KITTIDetectionDataset(args.dset_dir, args.name_pth, compact_class=args.kitti_compact)

    return dset


def get_detector_output(args, predict):
    bbox = predict[..., :4]
    conf = predict[..., 4]
    cls = torch.argmax(predict[..., 5:], dim=-1)

    conf_mask = conf >= args.conf_thresh
    bbox = bbox[conf_mask]
    conf = conf[conf_mask]
    cls = cls[conf_mask]

    bbox = convert_box_from_yxhw_to_xyxy(bbox) * 32

    bbox_per_class = []
    conf_per_class = []
    cls_per_class = []

    for i in range(1, args.num_classes):
        cls_idx = cls == i
        mask = nms(bbox[cls_idx], conf[cls_idx], args.nms_thresh)
        bbox_nms = bbox[cls_idx][mask]
        conf_nms = conf[cls_idx][mask]
        cls_nms = cls[cls_idx][mask]

        bbox_per_class.append(bbox_nms)
        conf_per_class.append(conf_nms)
        cls_per_class.append(cls_nms)

    bbox = torch.cat([*bbox_per_class], dim=0).type(torch.int)
    conf = torch.cat([*conf_per_class], dim=0)
    cls = torch.cat([*cls_per_class], dim=0)

    return bbox, conf, cls


def inference_detect(args, model):
    model.mode = 'detect'
    colors = [(127, 127, 0), (127, 0, 127), (0, 127, 127)]
    names = ['Person', 'Vehicle']

    with open(args.category_name, 'r') as f:
        category_name = list(map(str.strip, f.readlines()))

    if len(args.sample_pth) == 0:
        dset = get_dataset(args)
        for i in range(len(dset)):
            ann = dset[i]
            img = ann['img_left'].unsqueeze(0).to(args.device)
            img_np = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            tar_bbox = ann['bbox']

            pred = model(img)
            box_per_cls_list, conf_per_cls_list = model.detector.final_result(pred, args.conf_thresh, args.nms_thresh)

            for j in range(len(box_per_cls_list)):
                bbox = box_per_cls_list[j]
                conf = conf_per_cls_list[j]

                if len(bbox) > 0:
                    bbox[..., 0:3:2] *= args.img_size[1]
                    bbox[..., 1:4:2] *= args.img_size[0]

                    for m in range(len(bbox)):
                        x1, y1, x2, y2 = bbox[i]
                        img_np = cv.rectangle(img_np.copy(), (x1, y1), (x2, y2), (255, 0, 0), 2)

                    for m in range(len(bbox)):
                        x1, y1, x2, y2 = bbox[i]
                        conf_str = f'{conf[m]:.2f}'
                        name_str = f'{names[j]}'
                        font_scale = .3
                        font_thickness = 1

                        conf_size, _ = cv.getTextSize(conf_str, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        name_size, _ = cv.getTextSize(name_str, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

                        img_np = cv.rectangle(img_np.copy(), (x1, y1), (x1+conf_size[0], y1+conf_size[1]), colors[j], -1)
                        img_np = cv.putText(img_np.copy(), conf_str, (x1, y1+conf_size[1]), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

                        img_np = cv.rectangle(img_np.copy(), (x1, y1-name_size[1]), (x1 + conf_size[0], y1), colors[j], -1)
                        img_np = cv.putText(img_np.copy(), conf_str, (x1, y1-name_size[1]), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

                    # for i in range(len(tar_bbox)):
                    #     y1, x1, y2, x2 = list(map(int, tar_bbox[i]))
                    #     img_np = cv.rectangle(img_np.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)

            plt.imshow(img_np)
            plt.show()

    else:
        h, w = args.img_size
        img = cv.imread(args.sample_pth)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (w, h))
        img = T.ToTensor()(img).unsqueeze(0).to(args.device)

        pred = model(img)
        box_per_cls_list, conf_per_cls_list = model.detector.final_result(pred, args.conf_thresh, args.nms_thresh)

        for i in range(len(box_per_cls_list)):
            bbox = box_per_cls_list[i]
            conf = conf_per_cls_list[i]

            if len(bbox) > 0:
                bbox[..., 0:3:2] *= args.img_size[1]
                bbox[..., 1:4:2] *= args.img_size[0]

                for m in range(len(bbox)):
                    x1, y1, x2, y2 = bbox[i]
                    img_np = cv.rectangle(img_np.copy(), (x1, y1), (x2, y2), (255, 0, 0), 2)

                for m in range(len(bbox)):
                    x1, y1, x2, y2 = bbox[i]
                    conf_str = f'{conf[m]:.2f}'
                    name_str = f'{names[j]}'
                    font_scale = .3
                    font_thickness = 1

                    conf_size, _ = cv.getTextSize(conf_str, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    name_size, _ = cv.getTextSize(name_str, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

                    img_np = cv.rectangle(img_np.copy(), (x1, y1), (x1 + conf_size[0], y1 + conf_size[1]), colors[j], -1)
                    img_np = cv.putText(img_np.copy(), conf_str, (x1, y1 + conf_size[1]), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

                    img_np = cv.rectangle(img_np.copy(), (x1, y1 - name_size[1]), (x1 + conf_size[0], y1), colors[j], -1)
                    img_np = cv.putText(img_np.copy(), conf_str, (x1, y1 - name_size[1]), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

                # for i in range(len(tar_bbox)):
                #     y1, x1, y2, x2 = list(map(int, tar_bbox[i]))
                #     img_np = cv.rectangle(img_np.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)

        plt.imshow(img_np)
        plt.show()


def inference_depth(args, model):
    def disparity_to_depth(disparity):
        disp = T.Resize((372, 1242))(disparity)
        baseline = .54
        focal = 721

        return baseline * focal / (1242 * disp)

    model.mode = 'depth'

    if len(args.sample_pth) > 0:
        h, w = args.img_size
        img = cv.imread(args.sample_pth)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (w, h))
        img = T.ToTensor()(img).unsqueeze(0).to(args.device)
        disp = model(img)[0].squeeze(0)
        depth = disparity_to_depth(disp)
        depth_l = depth[0].detach().cpu().numpy()
        plt.imshow(depth_l)
        plt.show()
    else:
        pass


def inference(args, model):
    h, w = args.img_size
    with open(args.name_pth, 'r') as f_name:
        names = f_name.readlines()
    names = list(map(str.strip, names))
    colors = [(0, 0, 255), (255, 0, 0)]

    if len(args.sample_pth) == 0:
        pass
    else:
        img = cv.imread(args.sample_pth)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h_og, w_og, _ = img.shape
        img = cv.resize(img, (w, h))
        img_tensor = T.ToTensor()(img).unsqueeze(0).to(args.device)
        det, disp = model(img_tensor)
        disp = disp[0]

        bbox, conf, cls = model.detector.get_output(det, args.conf_thresh, args.nms_thresh)
        dr, dl = model.get_depth(disp)
        final_depth = dr

        num_obj = len(bbox)
        bbox[..., 0:3:2] *= w_og / w
        bbox[..., 1:4:2] *= h_og / h

        img = cv.resize(img, (w_og, h_og))
        for i in range(num_obj):
            x1, y1, x2, y2 = list(map(int, bbox[i]))
            ctr_x = int((x1 + x2) / 2)
            ctr_y = int((y1 + y2) / 2)
            depth = final_depth[ctr_y, ctr_x]
            cv.rectangle(img, (x1, y1), (x2, y2), colors[cls[i]], 2)

            conf_str = f'{conf[i]:.2f}'
            conf_size, _ = cv.getTextSize(conf_str, cv.FONT_HERSHEY_SIMPLEX, .5, 1)
            cv.rectangle(img, (x1, y1-conf_size[1]), (x1+conf_size[0], y1), colors[cls[i]], -1)
            cv.putText(img, conf_str, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

            cls_str = f'{names[cls[i]]}'
            cls_size, _ = cv.getTextSize(cls_str, cv.FONT_HERSHEY_SIMPLEX, .5, 1)
            cv.rectangle(img, (x1, y2), (x1+cls_size[0], y2+cls_size[1]), colors[cls[i]], -1)
            cv.putText(img, cls_str, (x1, y2+cls_size[1]), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

            depth_str = f'{depth:.2f}m'
            depth_size, _ = cv.getTextSize(depth_str, cv.FONT_HERSHEY_SIMPLEX, .5, 1)
            cv.rectangle(img, (x2-depth_size[0], y1), (x2, y1+depth_size[1]), colors[cls[i]], -1)
            cv.putText(img, depth_str, (x2-depth_size[0], y1+depth_size[1]), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        plt.subplot(211)
        plt.imshow(img)
        plt.subplot(212)
        plt.imshow(final_depth, vmax=80)
        plt.show()


def main(args):
    if args.model_name == 'udnet6':
        model = UDNet6(args.img_size, args.num_classes).to(torch.device(args.device))
    elif args.model_name == 'udnet7':
        model = UDNet7(args.img_size, args.num_classes).to(torch.device(args.device))
    elif args.model_name == 'udnet8':
        model = UDNet8(args.img_size, args.num_classes).to(torch.device(args.device))
    elif args.model_name == 'udnet9':
        model = UDNet9(args.img_size, args.num_classes).to(torch.device(args.device))
    else:
        print('Model name error')
        exit()
    model.load_state_dict(torch.load(args.weight_pth, map_location=torch.device(args.device))['model_state_dict'])
    model.eval()
    print('Model loaded.')

    net_mode = args.weight_pth.strip().split('/')[-1].split('_')[0]

    if net_mode == 'detect':
        inference_detect(args, model)
    elif net_mode == 'depth':
        inference_depth(args, model)
    else:
        inference(args, model)


def show_image(img):
    plt.imshow(img)
    plt.show()


def show_depth(imgs):
    num_img = len(imgs)
    cmap = cm.get_cmap('magma_r')
    for i in range(num_img):
        plot_idx = 100*num_img + 11 + i
        plt.subplot(plot_idx)
        plt.imshow(imgs[i], vmax=30, cmap=cmap)
    plt.show()


if __name__ == '__main__':
    def bool_type(str):
        if str == 'True':
            return True
        elif str == 'False':
            return False
        else:
            raise argparse.ArgumentTypeError('Not Boolean Value.')

    parser = argparse.ArgumentParser()

    model_name = 'udnet9'

    if model_name == 'udnet6':
        weight_pth = './weights/UDNet6_YOLOV3KITTIDataset_10epoch_0.000100_2.743loss(train)_2.759loss.ckpt'
    elif model_name == 'udnet7':
        weight_pth = './weights/UDNet7_YOLOV3KITTIDataset_22epoch_0.000010_2.120loss(train)_2.899loss.ckpt'
    elif model_name == 'udnet8':
        weight_pth = './weights/UDNet8_YOLOV3KITTIDataset_50epoch_0.000010_1.733loss(train)_2.269loss.ckpt'
    elif model_name == 'udnet9':
        weight_pth = './weights/(2nd)UDNet9_YOLOV3KITTIDataset_50epoch_0.000100_3.842loss(train)_4.057loss.ckpt'
        weight_pth = './weights/(2nd)UDNet9_YOLOV3KITTIDataset_25epoch_0.000001_1.045loss(train)_1.827loss.ckpt'
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    parser.add_argument('--device', required=False, type=str, default=device)
    parser.add_argument('--model_name', required=False, type=str, default=model_name)
    parser.add_argument('--weight_pth', required=False, type=str, default=weight_pth)
    parser.add_argument('--sample_pth', required=False, type=str, default='./sample/kitti_1.png')
    parser.add_argument('--name_pth', required=False, type=str, default='./dataset/kitti_compact.name')
    parser.add_argument('--dset_dir', required=False, type=str, default='D://DeepLearningData/KITTI/')
    parser.add_argument('--img_size', required=False, type=tuple, default=(256, 512))
    parser.add_argument('--num_classes', required=False, type=int, default=2)
    parser.add_argument('--conf_thresh', required=False, type=float, default=.5)
    parser.add_argument('--nms_thresh', required=False, type=float, default=.3)
    parser.add_argument('--category_name', required=False, type=str, default='./dataset/kitti.name')
    parser.add_argument('--kitti_compact', required=False, type=bool_type, default=True)

    args = parser.parse_args()

    main(args)