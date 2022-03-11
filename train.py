import os
import time
import datetime
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import ConcatDataset, DataLoader

from dataset.yolov3_kitti_dataset import YOLOV3KITTIDataset, custom_collate_fn
from utils.pytorch_util import make_batch, convert_box_from_yxhw_to_xyxy
from utils.util import time_calculator

from models.model9.udnet9 import UDNet9


def get_dataset(args):
    augmentation = None
    train_dset = YOLOV3KITTIDataset(
        root=args.dset_root,
        img_size=args.in_size,
        names_path=args.class_name_path,
        for_train=True,
        normalize=False,
        compact_class=args.compact_kitti,
        class_idx=0
    )
    train_dset = ConcatDataset([train_dset for _ in range(3)])
    val_dset = YOLOV3KITTIDataset(
        root=args.dset_root,
        img_size=args.in_size,
        names_path=args.class_name_path,
        for_train=False,
        normalize=False,
        compact_class=args.compact_kitti,
        class_idx=0
    )

    return val_dset.__class__.__name__, train_dset, val_dset, augmentation


def adjust_learning_rate(optimizer, current_epoch):
    if isinstance(optimizer, optim.Adam):
        if current_epoch == 0:
            optimizer.param_groups[0]['lr'] = .000001
    elif isinstance(optimizer, optim.SGD):
        if current_epoch == 0:
            optimizer.param_groups[0]['lr'] = .001


class NanError(Exception):
    def __init__(self, message='NaN appears. Don\'t backpropagate this time.'):
        self.message = message
        super().__init__(message)

class InfError(Exception):
    def __init__(self, message='Inf appears. Don\'t backpropagate thie time.'):
        self.message = message
        super().__init__(message)


def main(args):
    if not os.path.exists(args.record_dir):
        os.mkdir(args.record_dir)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.graph_dir):
        os.mkdir(args.graph_dir)

    dset_name, train_dset, val_dset, augmentation = get_dataset(args)

    model = UDNet9(
        args.in_size,
        args.num_classes
    ).to(args.device)
    model_name = model.__class__.__name__

    if args.optimizer == 'Adam' or 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd' or 'SGD' or 'Sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )

    now = datetime.datetime.now()
    date_str = f'{now.date().year}{now.date().month:02d}{now.date().day:02d}'
    time_str = f'{now.time().hour:02d}{now.time().minute:02d}{now.time().second:02d}'
    record_fn = f'record_{model_name}_{date_str}_{time_str}.csv'

    if not args.debug and args.save_graph:
        graph_dir = os.path.join(args.graph_dir, model_name)
        if not os.path.exists(graph_dir):
            os.mkdir(graph_dir)

    if len(args.ckpt_pth) > 0:
        ckpt = torch.load(args.ckpt_pth)
        model.load_state_dict(ckpt['model_state_dict'])
        print('Model state dict loaded.')
        if isinstance(optimizer, optim.Adam):
            if 'optimizer_state_dict' in ckpt.keys():
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print('Optimizer state dict loaded.')
        record_fn = 'transfer_' + record_fn

    record_dir = os.path.join(args.record_dir, model_name)
    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    record_pth = os.path.join(record_dir, record_fn)
    if not args.debug and args.save_record:
        with open(record_pth, 'w') as record:
            record.write('optimizer, epoch, lr, loss(train), loss, loss(detector), loss(box), loss(obj), loss(no_obj), loss(depth), iou, acc(cls)\n')

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    train_loss_list = []
    train_loss_detector_list = []
    train_loss_depth_list = []
    train_iou_list = []

    val_loss_list = []
    val_loss_detector_list = []
    val_loss_depth_list = []
    val_iou_list = []

    t_start = time.time()

    for e in range(args.epoch):
        train_loss = 0
        train_loss_detector = 0
        train_loss_box = 0
        train_loss_obj = 0
        train_loss_noobj = 0
        train_loss_depth = 0
        train_iou = 0
        train_acc_conf = 0
        train_acc_cls = 0

        val_loss = 0
        val_loss_detector = 0
        val_loss_box = 0
        val_loss_obj = 0
        val_loss_noobj = 0
        val_loss_depth = 0
        val_iou = 0
        val_acc_conf = 0
        val_acc_cls = 0

        pred_box_list = []
        tar_box_list = []
        pred_label_list = []
        tar_label_list = []

        adjust_learning_rate(optimizer, e)
        cur_lr = optimizer.param_groups[0]['lr']
        t_train_start = time.time()

        num_batch = 0
        num_data = 0

        t_fold_start = time.time()
        model.train()
        for i, ann in enumerate(train_loader):
            num_batch += 1
            num_data += len(ann)

            print(f'[{e+1}/{args.epoch}] ', end='')
            print(f'{num_data}/{len(train_dset)} ', end='')
            print(f'<lr> {cur_lr:.6f}  ', end='')

            imgl = make_batch(ann, 'imgl').to(args.device)
            imgr = make_batch(ann, 'imgr').to(args.device)
            tar1 = make_batch(ann, 'target1').to(args.device)
            tar2 = make_batch(ann, 'target2').to(args.device)
            tar3 = make_batch(ann, 'target3').to(args.device)
            tar_detector = [tar1, tar2, tar3]

            pred_detector, pred_disp = model(imgl)

            loss, loss_detector, loss_box, loss_obj, loss_noobj, _, loss_depth, iou, acc_conf, acc_cls = model.loss(pred_detector, tar_detector, imgl, imgr, pred_disp)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            loss = loss.detach().cpu().item()

            train_loss += loss
            train_loss_detector += loss_detector
            train_loss_box += loss_box
            train_loss_obj += loss_obj
            train_loss_noobj += loss_noobj
            train_loss_depth += loss_depth
            train_iou += iou
            train_acc_conf += acc_conf
            train_acc_cls += acc_cls

            t_iter_end = time.time()
            h, m, s = time_calculator(t_iter_end - t_start)

            print(f'<loss> {loss:.4f} ({train_loss/num_batch:.4f})  ', end='')
            print(f'<loss_detector> {loss_detector:.4f} ({train_loss_detector/num_batch:.4f})  ', end='')
            print(f'<loss_depth> {loss_depth:.4f} ({train_loss_depth/num_batch:.4f})  ', end='')
            print(f'<iou> {iou:.4f} ({train_iou/num_batch:.4f})  ', end='')
            print(f'<acc_cls> {acc_cls:.4f} ({train_acc_cls/num_batch:.4f})  ', end='')
            print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}', end='')

            if num_data == len(train_dset):
                print()
            else:
                print(end='\r')

            # import cv2 as cv
            # import numpy as np
            # from utils.pytorch_util import convert_box_from_yxhw_to_xyxy
            # plt.figure(e)
            # for j in range(batch_size):
            #     plt.subplot(220+j+1)
            #     img_np = img_left[j].permute(1, 2, 0).detach().cpu().numpy()
            #     bbox = ann[j]['bbox']
            #     bbox_np = bbox.detach().cpu().numpy()
            #     bbox_np[..., 0:3:2] *= 256 / 8
            #     bbox_np[..., 1:4:2] *= 512 / 16
            #     bbox_np = bbox_np.astype(np.int)
            #     for b in bbox_np:
            #         y1, x1, y2, x2 = b
            #         img_np = cv.rectangle(img_np.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     plt.imshow(img_np)
            # plt.show()

            del imgl, imgr, tar1, tar2, tar3, tar_detector, pred_detector, pred_disp, loss, loss_detector, loss_depth, iou
            torch.cuda.empty_cache()

            if args.debug:
                if i == 5:
                    break

        train_loss /= num_batch
        train_loss_detector /= num_batch
        train_loss_box /= num_batch
        train_loss_obj /= num_batch
        train_loss_noobj /= num_batch
        train_loss_depth /= num_batch
        train_iou /= num_batch
        train_acc_conf /= num_batch
        train_acc_cls /= num_batch

        t_train_end = time.time()
        h, m, s = time_calculator(t_train_end - t_fold_start)

        print(f'[{e+1}/{args.epoch}]\t\t(train) - ', end='')
        print(f'<loss> {train_loss:.3f}  ', end='')
        print(f'<loss_detector> {train_loss_detector:.3f}  ', end='')
        print(f'<loss_box> {train_loss_box:.3f}  ', end='')
        print(f'<loss_obj> {train_loss_obj:.3f}  ', end='')
        print(f'<loss_noobj> {train_loss_noobj:.3f}  ', end='')
        print(f'<loss_depth> {train_loss_depth:.3f}  ', end='')
        print(f'<iou> {train_iou:.3f}  ', end='')
        print(f'<acc_cls> {train_acc_cls:.3f}  ', end='')
        print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

        with torch.no_grad():
            num_batch = 0
            num_data = 0
            pred_box_list = []
            tar_box_list = []
            pred_label_list = []
            tar_label_list = []
            t_val_start = time.time()

            model.eval()
            for i, ann in enumerate(val_loader):
                num_batch += 1
                num_data += len(ann)

                imgl = make_batch(ann, 'imgl').to(args.device)
                imgr = make_batch(ann, 'imgr').to(args.device)
                tar1 = make_batch(ann, 'target1').to(args.device)
                tar2 = make_batch(ann, 'target2').to(args.device)
                tar3 = make_batch(ann, 'target3').to(args.device)
                tar_detector = [tar1, tar2, tar3]
                # tar_box = convert_box_from_yxhw_to_xyxy(ann['bbox'])
                # tar_label = ann['label']

                pred_detector, pred_disp = model(imgl)

                loss, loss_detector, loss_box, loss_obj, loss_noobj, _, loss_depth, iou, acc_conf, acc_cls = model.loss(pred_detector, tar_detector, imgl, imgr, pred_disp)
                loss = loss.detach().cpu().item()

                # pred_box, pred_conf, pred_cls = model.detector.get_output(pred_detector, args.conf_thresh, args.nms_thresh)
                # pred_box_list.append(pred_box.detach().cpu())
                # pred_label_list.append(pred_cls.detach().cpu())
                # tar_box_list.append(tar_box)
                # tar_label_list.append(tar_label)



                val_loss += loss
                val_loss_detector += loss_detector
                val_loss_box += loss_box
                val_loss_obj += loss_obj
                val_loss_noobj += loss_noobj
                val_loss_depth += loss_depth
                val_iou += iou
                val_acc_conf += acc_conf
                val_acc_cls += acc_cls

                del imgl, imgr, tar1, tar2, tar3, tar_detector, pred_detector, pred_disp, loss, loss_detector, loss_depth, iou
                torch.cuda.empty_cache()

                print(f'Validating {num_data}/{len(val_dset)}', end='')
                print(end='\r')

                if args.debug:
                    if i == 5:
                        break

        val_loss /= num_batch
        val_loss_detector /= num_batch
        val_loss_box /= num_batch
        val_loss_obj /= num_batch
        val_loss_noobj /= num_batch
        val_loss_depth /= num_batch
        val_iou /= num_batch
        val_acc_conf /= num_batch
        val_acc_cls /= num_batch

        t_val_end = time.time()

        h, m, s = time_calculator(t_val_end - t_val_start)

        print(f'[{e+1}/{args.epoch}]\t\t(valid) - ', end='')
        print(f'<loss> {val_loss:.3f}  ', end='')
        print(f'<loss_detector> {val_loss_detector:.3f}  ', end='')
        print(f'<loss_box> {val_loss_box:.3f}  ', end='')
        print(f'<loss_obj> {val_loss_obj:.3f}  ', end='')
        print(f'<loss_noobj> {val_loss_noobj:.3f}  ', end='')
        print(f'<loss_depth> {val_loss_depth:.3f}  ', end='')
        print(f'<iou> {val_iou:.3f}  ', end='')
        print(f'<acc_cls> {val_acc_cls:.3f}  ', end='')
        print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

        train_loss_list.append(train_loss)
        train_loss_detector_list.append(train_loss_detector)
        train_loss_depth_list.append(train_loss_depth)
        train_iou_list.append(train_iou)

        val_loss_list.append(val_loss)
        val_loss_detector_list.append(val_loss_detector)
        val_loss_depth_list.append(val_loss_depth)
        val_iou_list.append(val_iou)

        if not args.debug and args.save_record:
            with open(record_pth, 'a') as record:
                record_str = f'{args.optimizer}, ' \
                             f'{e + 1}, ' \
                             f'{cur_lr}, ' \
                             f'{train_loss:.3f}, ' \
                             f'{val_loss:.3f}, ' \
                             f'{val_loss_detector:.3f}, ' \
                             f'{val_loss_box:.3f}, ' \
                             f'{val_loss_obj:.3f}, ' \
                             f'{val_loss_noobj:.3f}, ' \
                             f'{val_loss_depth:.3f}, ' \
                             f'{val_iou:.3f}, ' \
                             f'{val_acc_cls:.3f}\n'
                record.write(record_str)

        if args.debug:
            save_dict = {}
            save_dict['model_state_dict'] = model.state_dict()
            if isinstance(optimizer, optim.Adam):
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_fn = 'test.ckpt'
            save_pth = os.path.join(args.save_dir, save_fn)
            torch.save(save_dict, save_pth)

        if not args.debug and e % args.save_term == 0:
            save_dict = {}
            save_dict['model_state_dict'] = model.state_dict()
            if isinstance(optimizer, optim.Adam):
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_fn = f'(2nd){model_name}_' \
                      f'{dset_name}_' \
                      f'{e+1}epoch_' \
                      f'{cur_lr:.6f}_' \
                      f'{train_loss:.3f}loss(train)_' \
                      f'{val_loss:.3f}loss_' \
                      f'{val_loss_detector:.3f}loss(detector)_' \
                      f'{val_loss_depth:.3f}loss(depth)_' \
                      f'{val_iou:.3f}iou_' \
                      f'{val_acc_cls:.3f}acc(cls).ckpt'
            save_pth = os.path.join(args.save_dir, save_fn)
            torch.save(save_dict, save_pth)

        if not args.debug and args.save_graph:
            loss_graph_pth = os.path.join(graph_dir, f'loss_{model_name}_{date_str}_{time_str}.png')
            iou_graph_pth = os.path.join(graph_dir, f'iou_{model_name}_{date_str}_{time_str}.png')
            x_axis = [i for i in range(len(train_loss_list))]

            plt.figure(e)
            plt.plot(x_axis, train_loss_list, 'r-', label='Train')
            plt.plot(x_axis, val_loss_list, 'b-', label='Val')
            plt.title('Loss')
            plt.legend()
            plt.savefig(loss_graph_pth)
            plt.close()

            plt.figure(e)
            plt.plot(x_axis, train_iou_list, 'r-', label='Train')
            plt.plot(x_axis, val_iou_list, 'b-', label='Val')
            plt.title('IoU')
            plt.legend()
            plt.savefig(iou_graph_pth)
            plt.close()


if __name__ == '__main__':
    def type_bool(value):
        if value == 'True':
            return True
        elif value == 'False':
            return False
        else:
            raise argparse.ArgumentTypeError('Not Boolean value.')

    parser = argparse.ArgumentParser()

    ckpt_pth = ''
    ckpt_pth = './weights/(2nd)UDNet9_YOLOV3KITTIDataset_46epoch_0.000010_1.092loss(train)_1.793loss.ckpt'

    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--lr', type=float, required=False, default=.0001)
    parser.add_argument('--weight_decay', type=float, required=False, default=.0005)
    parser.add_argument('--momentum', type=float, required=False, default=.9)
    parser.add_argument('--epoch', type=int, required=False, default=50)
    parser.add_argument('--dset_root', type=str, required=False, default='D://DeepLearningData/KITTI')
    # parser.add_argument('--model_ckpt_pth', type=str, required=False, default='./weights/ObjectDepthNet3_YOLOV3KITTIDataset_20epoch_0.000100_2.285loss(train)_2.706loss_1.006loss(detector)_1.699loss(depth)_0.527iou.ckpt')    # Blank if none
    parser.add_argument('--ckpt_pth', type=str, required=False, default=ckpt_pth)    # Blank if none
    parser.add_argument('--class_name_path', type=str, required=False, default='./dataset/kitti_compact.name')
    parser.add_argument('--in_size', type=tuple, required=False, default=(256, 512))
    parser.add_argument('--num_classes', type=int, required=False, default=2)    # KITTI objects
    parser.add_argument('--optimizer', type=str, required=False, default='Adam')    # Adam or SGD
    parser.add_argument('--save_dir', type=str, required=False, default='./save/')
    parser.add_argument('--record_dir', type=str, required=False, default='./records/')
    parser.add_argument('--graph_dir', type=str, required=False, default='./graph/')
    parser.add_argument('--save_graph', type=type_bool, required=False, default=True)
    parser.add_argument('--save_record', type=type_bool, required=False, default=True)
    parser.add_argument('--compact_kitti', type=type_bool, required=False, default=True)
    parser.add_argument('--save_term', type=int, required=False, default=1)
    parser.add_argument('--debug', type=type_bool, required=False, default=False)

    args = parser.parse_args()

    torch.set_num_threads(1)
    main(args)