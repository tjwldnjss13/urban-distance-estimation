import os
import time
import datetime
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold

from dataset.yolov3_kitti_dataset import YOLOV3KITTIDataset, custom_collate_fn
from dataset.augment import rotate2d_augmentation, shift_augmentation, horizontal_flip_augmentation
from utils.pytorch_util import make_batch
from utils.util import time_calculator

from models.model3.object_depth_net3 import ObjectDepthNet3


def get_dataset(args):
    augmentation = [rotate2d_augmentation, shift_augmentation, horizontal_flip_augmentation]
    dset = YOLOV3KITTIDataset(root=args.dset_root, names_path=args.class_name_path, normalize=False, compact_class=args.compact_kitti)

    return dset.__class__.__name__, dset, augmentation


def adjust_learning_rate(optimizer, current_epoch):
    if isinstance(optimizer, optim.Adam):
        if current_epoch == 0:
            optimizer.param_groups[0]['lr'] = .0001
        elif current_epoch == 5:
            optimizer.param_groups[0]['lr'] = .0001
        elif current_epoch == 50:
            optimizer.param_groups[0]['lr'] = .00001
        elif current_epoch == 130:
            optimizer.param_groups[0]['lr'] = .000001
    elif isinstance(optimizer, optim.SGD):
        if current_epoch == 0:
            optimizer.param_groups[0]['lr'] = .001
        elif current_epoch == 30:
            optimizer.param_groups[0]['lr'] = .001
        elif current_epoch == 40:
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
    device = args.device
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    momentum = args.momentum
    epoch = args.epoch
    dset_root = args.dset_root
    model_ckpt_pth = args.model_ckpt_pth
    in_size = args.in_size
    num_classes = args.num_classes
    optim_name = args.optimizer
    save_dir = args.save_dir
    record_dir = args.record_dir
    graph_dir = args.graph_dir

    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)

    dset_name, dset, augmentation = get_dataset(args)
    anchors = dset.anchors.to(device)

    model = ObjectDepthNet3(in_size, num_classes, mode='detect').to(device)
    model_name = model.__class__.__name__

    if optim_name == 'Adam' or 'adam':
        optimizer = optim.Adam(params=model.parameters(), weight_decay=weight_decay)
    elif optim_name == 'sgd' or 'SGD' or 'Sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=.001, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    now = datetime.datetime.now()
    date_str = f'{now.date().year}{now.date().month:02d}{now.date().day:02d}'
    time_str = f'{now.time().hour:02d}{now.time().minute:02d}{now.time().second:02d}'
    record_fn = f'record_{model_name}_{date_str}_{time_str}.csv'

    if len(model_ckpt_pth) > 0:
        ckpt = torch.load(model_ckpt_pth)
        model.load_state_dict(ckpt['model_state_dict'])
        print('Model state dict loaded.')
        if isinstance(optimizer, optim.Adam):
            if 'optimizer_state_dict' in ckpt.keys():
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print('Optimizer state dict loaded.')
        record_fn = 'transfer_' + record_fn

    record_pth = os.path.join(record_dir, record_fn)
    if args.save_record:
        with open(record_pth, 'w') as record:
            record.write('optimizer, epoch, lr, loss(train), loss,loss(box), loss(obj), loss(no_obj), loss(cls)\n')

    kfold = KFold(n_splits=args.num_k_fold, shuffle=True)

    train_loss_list = []
    train_iou_list = []
    val_loss_list = []
    val_iou_list = []

    t_start = time.time()

    for e in range(epoch):
        if e < 5:
            model.detector.train_safe = True
        else:
            model.detector.train_safe = False

        num_batch = 0
        num_data = 0

        train_loss = 0
        train_iou = 0

        val_loss = 0
        val_loss_box = 0
        val_loss_obj = 0
        val_loss_no_obj = 0
        val_loss_cls = 0
        val_iou = 0

        adjust_learning_rate(optimizer, e)
        cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        t_train_start = time.time()

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dset)):
            train_dset = Subset(dset, train_idx)
            train_dset.dataset.augmentation = augmentation
            val_dset = Subset(dset, val_idx)

            train_loader = DataLoader(train_dset, batch_size=batch_size, collate_fn=custom_collate_fn)
            val_loader = DataLoader(val_dset, batch_size=batch_size, collate_fn=custom_collate_fn)

            train_loss_fold = 0
            train_loss_box_fold = 0
            train_loss_obj_fold = 0
            train_loss_no_obj_fold = 0
            train_loss_cls_fold = 0
            train_iou_fold = 0

            val_loss_fold = 0
            val_loss_box_fold = 0
            val_loss_obj_fold = 0
            val_loss_no_obj_fold = 0
            val_loss_cls_fold = 0
            val_iou_fold = 0

            num_batch = 0
            num_data = 0

            t_fold_start = time.time()

            for i, ann in enumerate(train_loader):
                num_batch += 1
                num_data += len(ann)

                print(f'[{e+1}/{epoch} ({fold+1}/{args.num_k_fold})] ', end='')
                print(f'{num_data}/{len(train_dset)} ', end='')
                print(f'<lr> {cur_lr:.6f}  ', end='')

                img = make_batch(ann, 'img').to(device)
                tar1 = make_batch(ann, 'target1').to(device)
                tar2 = make_batch(ann, 'target2').to(device)
                tar3 = make_batch(ann, 'target3').to(device)

                pred1, pred2, pred3 = model(img)

                try:
                    loss1, loss_box1, loss_obj1, loss_no_obj1, loss_cls1, iou1 = model.loss_detector(pred1, tar1)
                    loss2, loss_box2, loss_obj2, loss_no_obj2, loss_cls2, iou2 = model.loss_detector(pred2, tar2)
                    loss3, loss_box3, loss_obj3, loss_no_obj3, loss_cls3, iou3 = model.loss_detector(pred3, tar3)

                    loss = (loss1 + loss2 + loss3) / 3
                    loss_box = (loss_box1 + loss_box2 + loss_box3) / 3
                    loss_obj = (loss_obj1 + loss_obj2 + loss_obj3) / 3
                    loss_no_obj = (loss_no_obj1 + loss_no_obj2 + loss_no_obj3) / 3
                    loss_cls = (loss_cls1 + loss_cls2 + loss_cls3) / 3
                    iou = (iou1 + iou2 + iou3) / 3

                    if torch.isnan(loss):
                        raise NanError
                    elif torch.isinf(loss):
                        raise InfError
                except NanError:
                    print()
                except InfError:
                    print()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss = loss.detach().cpu().item()

                train_loss_fold += loss
                train_loss_box_fold += loss_box
                train_loss_obj_fold += loss_obj
                train_loss_no_obj_fold += loss_no_obj
                train_loss_cls_fold += loss_cls
                train_iou_fold += iou

                t_iter_end = time.time()
                h, m, s = time_calculator(t_iter_end - t_start)

                print(f'<loss> {loss:.4f} ({train_loss_fold/num_batch:.4f})  ', end='')
                print(f'<loss_box> {loss_box:.4f} ({train_loss_box_fold/num_batch:.4f})  ', end='')
                print(f'<loss_obj> {loss_obj:.4f} ({train_loss_obj_fold/num_batch:.4f})  ', end='')
                print(f'<loss_no_obj> {loss_no_obj:.4f} ({train_loss_no_obj_fold/num_batch:.4f})  ', end='')
                print(f'<loss_cls> {loss_cls:.4f} ({train_loss_cls_fold/num_batch:.4f})  ', end='')
                print(f'<iou> {iou:.4f} ({train_iou_fold/num_batch:.4f})  ', end='')
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

                del img, tar1, tar2, tar3, pred1, pred2, pred3, loss, loss1, loss2, loss3
                torch.cuda.empty_cache()

            train_loss_fold /= num_batch
            train_loss_box_fold /= num_batch
            train_loss_obj_fold /= num_batch
            train_loss_no_obj_fold /= num_batch
            train_loss_cls_fold /= num_batch
            train_iou_fold /= num_batch

            train_loss += train_loss_fold
            train_iou += train_iou_fold

            t_train_end = time.time()
            h, m, s = time_calculator(t_train_end - t_fold_start)

            print('\t\t(train) - ', end='')
            print(f'<loss> {train_loss_fold:.4f}  ', end='')
            print(f'<loss_box> {train_loss_box_fold:.4f}  ', end='')
            print(f'<loss_obj> {train_loss_obj_fold:.4f}  ', end='')
            print(f'<loss_no_obj> {train_loss_no_obj_fold:.4f}  ', end='')
            print(f'<loss_cls> {train_loss_cls_fold:.4f}  ', end='')
            print(f'<iou> {train_iou_fold:.4f}  ', end='')
            print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

            model.eval()
            num_batch = 0
            num_data = 0
            t_val_start = time.time()

            for i, ann in enumerate(val_loader):
                num_batch += 1
                num_data += len(ann)

                img = make_batch(ann, 'img').to(device)
                tar1 = make_batch(ann, 'target1').to(device)
                tar2 = make_batch(ann, 'target2').to(device)
                tar3 = make_batch(ann, 'target3').to(device)

                pred1, pred2, pred3 = model(img)

                loss1, loss_box1, loss_obj1, loss_no_obj1, loss_cls1, iou1 = model.loss_detector(pred1, tar1)
                loss2, loss_box2, loss_obj2, loss_no_obj2, loss_cls2, iou2 = model.loss_detector(pred2, tar2)
                loss3, loss_box3, loss_obj3, loss_no_obj3, loss_cls3, iou3 = model.loss_detector(pred3, tar3)

                loss = (loss1 + loss2 + loss3) / 3
                loss_box = (loss_box1 + loss_box2 + loss_box3) / 3
                loss_obj = (loss_obj1 + loss_obj2 + loss_obj3) / 3
                loss_no_obj = (loss_no_obj1 + loss_no_obj2 + loss_no_obj3) / 3
                loss_cls = (loss_cls1 + loss_cls2 + loss_cls3) / 3
                iou = (iou1 + iou2 + iou3) / 3

                loss = loss.detach().cpu().item()

                val_loss_fold += loss
                val_loss_box_fold += loss_box
                val_loss_obj_fold += loss_obj
                val_loss_no_obj_fold += loss_no_obj
                val_loss_cls_fold += loss_cls
                val_iou_fold += iou

                del img, tar1, tar2, tar3, pred1, pred2, pred3, loss, loss1, loss2, loss3
                torch.cuda.empty_cache()

                print(f'Validating {num_data}/{len(val_dset)}', end='')
                print(end='\r')

            val_loss_fold /= num_batch
            val_loss_box_fold /= num_batch
            val_loss_obj_fold /= num_batch
            val_loss_no_obj_fold /= num_batch
            val_loss_cls_fold /= num_batch
            val_iou_fold /= num_batch

            val_loss += val_loss_fold
            val_loss_box += val_loss_box_fold
            val_loss_obj += val_loss_obj_fold
            val_loss_no_obj += val_loss_no_obj_fold
            val_loss_cls += val_loss_cls_fold
            val_iou += val_iou_fold

            t_val_end = time.time()

            h, m, s = time_calculator(t_val_end - t_val_start)

            print('\t\t(valid) - ', end='')
            print(f'<loss> {val_loss_fold:.5f}  ', end='')
            print(f'<loss_box> {val_loss_box_fold:.5f}  ', end='')
            print(f'<loss_obj> {val_loss_obj_fold:.5f}  ', end='')
            print(f'<loss_no_obj> {val_loss_no_obj_fold:.5f}  ', end='')
            print(f'<loss_cls> {val_loss_cls_fold:.5f}  ', end='')
            print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

        train_loss /= args.num_k_fold
        train_iou /= args.num_k_fold

        val_loss /= args.num_k_fold
        val_loss_box /= args.num_k_fold
        val_loss_obj /= args.num_k_fold
        val_loss_no_obj /= args.num_k_fold
        val_loss_cls /= args.num_k_fold
        val_iou /= args.num_k_fold

        train_loss_list.append(train_loss)
        train_iou_list.append(train_iou)
        val_loss_list.append(val_loss)
        val_iou_list.append(val_iou)

        print('----------------(kfold) - ', end='')
        print(f'<loss> {val_loss:.4f}  ', end='')
        print(f'<loss_box> {val_loss_box:.4f}  ', end='')
        print(f'<loss_obj> {val_loss_obj:.4f}  ', end='')
        print(f'<loss_no_obj> {val_loss_no_obj:.4f}  ', end='')
        print(f'<loss_cls> {val_loss_cls:.4f}  ', end='')
        print(f'<iou> {val_iou:.4f}')

        if args.save_record:
            with open(record_pth, 'a') as record:
                record_str = f'{optim_name}, ' \
                             f'{e + 1}, ' \
                             f'{cur_lr}, ' \
                             f'{train_loss:.4f}, ' \
                             f'{val_loss:.4f}, ' \
                             f'{val_loss_box:.4f}, ' \
                             f'{val_loss_obj:.4f}, ' \
                             f'{val_loss_no_obj:.4f}, ' \
                             f'{val_loss_cls:.4f}, ' \
                             f'{val_iou:.4f}\n'
                record.write(record_str)

        if e % args.save_term == 0:
            save_dict = {}
            save_dict['model_state_dict'] = model.state_dict()
            if isinstance(optimizer, optim.Adam):
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_pth = save_dir + f'detect_' \
                                  f'{model_name}_' \
                                  f'{dset_name}_' \
                                  f'{e+1}epoch_' \
                                  f'{cur_lr:.6f}_' \
                                  f'{train_loss:.3f}loss(train)_' \
                                  f'{val_loss:.3f}loss_' \
                                  f'{val_loss_box:.3f}loss(box)_' \
                                  f'{val_loss_obj:.3f}loss(obj)_' \
                                  f'{val_loss_no_obj:.3f}loss(noobj)_' \
                                  f'{val_loss_cls:.3f}loss(cls)_' \
                                  f'{val_iou:.3f}iou.ckpt'
            torch.save(save_dict, save_pth)

        if args.save_graph:
            loss_graph_pth = graph_dir + f'loss_{model_name}_{date_str}_{time_str}.png'
            iou_graph_pth = graph_dir + f'iou_{model_name}_{date_str}_{time_str}.png'
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

    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--weight_decay', type=float, required=False, default=.0005)
    parser.add_argument('--momentum', type=float, required=False, default=.9)
    parser.add_argument('--epoch', type=int, required=False, default=100)
    parser.add_argument('--dset_root', type=str, required=False, default='D://DeepLearningData/KITTI')
    # parser.add_argument('--model_ckpt_pth', type=str, required=False, default='./weights/detect/Adam_ObjectDepthNet2_YOLOKITTIDataset_22epoch_0.000100_1.074loss(train)_1.079loss_0.668loss(box)_0.099loss(obj)_0.089loss(noobj)_0.223loss(cls).ckpt')    # Blank if none
    parser.add_argument('--model_ckpt_pth', type=str, required=False, default='./weights/detect/detect_ObjectDepthNet3_YOLOV3KITTIDataset_29epoch_0.000100_1.309loss(train)_1.372loss.ckpt')    # Blank if none
    parser.add_argument('--class_name_path', type=str, required=False, default='./dataset/kitti_compact.name')
    parser.add_argument('--in_size', type=tuple, required=False, default=(256, 512))
    parser.add_argument('--num_classes', type=int, required=False, default=3)    # KITTI objects
    parser.add_argument('--optimizer', type=str, required=False, default='Adam')    # Adam or SGD
    parser.add_argument('--save_dir', type=str, required=False, default='./save/detect/')
    parser.add_argument('--record_dir', type=str, required=False, default='./records/detect/')
    parser.add_argument('--graph_dir', type=str, required=False, default='./graph/detect/')
    parser.add_argument('--save_graph', type=type_bool, required=False, default=True)
    parser.add_argument('--save_record', type=type_bool, required=False, default=True)
    parser.add_argument('--compact_kitti', type=type_bool, required=False, default=True)
    parser.add_argument('--num_k_fold', type=int, required=False, default=3)
    parser.add_argument('--save_term', type=int, required=False, default=1)

    args = parser.parse_args()

    torch.set_num_threads(1)
    main(args)