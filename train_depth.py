import os
import time
import datetime
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import random_split, DataLoader

from dataset.yolo_kitti_dataset import YOLOKITTIDataset, custom_collate_fn
from utils.pytorch_util import make_batch
from utils.util import time_calculator

from models.model1.object_depth_net import ObjectDepthNet
# from models.model2.object_depth_net2 import ObjectDepthNet2


def get_dataset(root):
    dset = YOLOKITTIDataset(root)

    num_train_dset = int(len(dset) * .7)
    num_val_dset = len(dset) - num_train_dset

    train_dset, val_dset = random_split(dset, [num_train_dset, num_val_dset])

    return dset.__class__.__name__, train_dset, val_dset


def adjust_learning_rate(optimizer, current_epoch):
    if isinstance(optimizer, optim.Adam):
        if current_epoch == 0:
            optimizer.param_groups[0]['lr'] = .0001
        elif current_epoch == 5:
            optimizer.param_groups[0]['lr'] = .0001
        elif current_epoch == 15:
            optimizer.param_groups[0]['lr'] = .0001
    elif isinstance(optimizer, optim.SGD):
        if current_epoch == 0:
            optimizer.param_groups[0]['lr'] = .001
        elif current_epoch == 30:
            optimizer.param_groups[0]['lr'] = .001
        elif current_epoch == 40:
            optimizer.param_groups[0]['lr'] = .001


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

    dset_name, train_dset, val_dset = get_dataset(dset_root)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dset, batch_size=batch_size, collate_fn=custom_collate_fn)

    model = ObjectDepthNet(in_size, num_classes, mode='depth').to(device)
    model_name = model.__class__.__name__

    if optim_name == 'Adam' or 'adam':
        optimizer = optim.Adam(params=model.parameters(), weight_decay=weight_decay)
    elif optim_name == 'sgd' or 'SGD' or 'Sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=.001, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    if len(model_ckpt_pth) > 0:
        ckpt = torch.load(model_ckpt_pth)
        model.load_state_dict(ckpt['model_state_dict'])
        if isinstance(optimizer, optim.Adam):
            optimizer.load_state_dict(ckpt['optimizer_stete_dict'])

    if not os.path.exists(record_dir):
        os.mkdir(record_dir)
    now = datetime.datetime.now()
    date_str = f'{now.date().year}{now.date().month:02d}{now.date().day:02d}'
    time_str = f'{now.time().hour:02d}{now.time().minute:02d}{now.time().second:02d}'
    record_pth = record_dir + f'record_{model_name}_{date_str}_{time_str}.csv'
    record = open(record_pth, 'w')
    record.write('optimizer, epoch, lr, loss(train), loss, loss(ap), loss(ds), loss(lr)\n')
    record.close()

    train_loss_list = []
    val_loss_list = []

    t_start = time.time()

    for e in range(epoch):
        num_batch = 0
        num_data = 0

        train_loss = 0
        train_loss_ap = 0
        train_loss_ds = 0
        train_loss_lr = 0

        val_loss = 0
        val_loss_ap = 0
        val_loss_ds = 0
        val_loss_lr = 0

        adjust_learning_rate(optimizer, e)
        cur_lr = optimizer.param_groups[0]['lr']

        model.train()

        t_train_start = time.time()

        for i, ann in enumerate(train_loader):
            num_batch += 1
            num_data += len(ann)

            print(f'[{e+1}/{epoch}] ', end='')
            print(f'{num_data}/{len(train_dset)} ', end='')
            print(f'<lr> {cur_lr:.6f}  ', end='')

            img_left = make_batch(ann, 'img_left').to(device)
            img_right = make_batch(ann, 'img_right').to(device)

            pred_disparity = model(img_left)

            loss, loss_ap, loss_ds, loss_lr = model.loss_depth(img_left, img_right, pred_disparity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().item()

            train_loss += loss
            train_loss_ap += loss_ap
            train_loss_ds += loss_ds
            train_loss_lr += loss_lr

            t_iter_end = time.time()
            h, m, s = time_calculator(t_iter_end - t_start)

            print(f'<loss> {loss:.5f} ({train_loss/num_batch:.5f})  ', end='')
            print(f'<loss_ap> {loss_ap:.5f} ({train_loss_ap/num_batch:.5f})  ', end='')
            print(f'<loss_ds> {loss_ds:.5f} ({train_loss_ds/num_batch:.5f})  ', end='')
            print(f'<loss_lr> {loss_lr:.5f} ({train_loss_lr/num_batch:.5f})  ', end='')
            print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

            del img_left, img_right, pred_disparity, loss
            torch.cuda.empty_cache()

        train_loss /= num_batch
        train_loss_ap /= num_batch
        train_loss_ds /= num_batch
        train_loss_lr /= num_batch

        train_loss_list.append(train_loss)

        t_train_end = time.time()
        h, m, s = time_calculator(t_train_end - t_train_start)

        print('\t\t', end='')
        print(f'<train_loss> {train_loss:.5f}  ', end='')
        print(f'<train_loss_ap> {train_loss_ap:.5f}  ', end='')
        print(f'<train_loss_ds> {train_loss_ds:.5f}  ', end='')
        print(f'<train_loss_lr> {train_loss_lr:.5f}  ', end='')
        print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

        print('Validating...')


        model.eval()
        num_batch = 0
        t_val_start = time.time()

        for i, ann in enumerate(val_loader):
            num_batch += 1

            img_left = make_batch(ann, 'img_left').to(device)
            img_right = make_batch(ann, 'img_right').to(device)

            pred_disparity = model(img_left)
            loss, loss_ap, loss_ds, loss_lr = model.loss_depth(img_left, img_right, pred_disparity)

            loss = loss.detach().cpu().item()

            val_loss += loss
            val_loss_ap += loss_ap
            val_loss_ds += loss_ds
            val_loss_lr += loss_lr

            del img_left, img_right, pred_disparity, loss
            torch.cuda.empty_cache()

        val_loss /= num_batch
        val_loss_ap /= num_batch
        val_loss_ds /= num_batch
        val_loss_lr /= num_batch

        val_loss_list.append(val_loss)

        t_val_end = time.time()

        h, m, s = time_calculator(t_val_end - t_val_start)

        print('\t\t', end='')
        print(f'<val_loss> {val_loss:.5f}  ', end='')
        print(f'<val_loss_ap> {val_loss_ap:.5f}  ', end='')
        print(f'<val_loss_ds> {val_loss_ds:.5f}  ', end='')
        print(f'<val_loss_lr> {val_loss_lr:.5f}  ', end='')
        print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

        record = open(record_pth, 'a')
        record_str = f'{optim_name}, {e + 1}, {cur_lr}, {train_loss:.5f}, {val_loss:.5f}, {val_loss_ap:.5f}, {val_loss_ds:.5f}, {val_loss_lr:.5f}\n'
        record.write(record_str)
        record.close()

        if (e + 1) % 5 == 0:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_dict = {}
            save_dict['model_state_dict'] = model.state_dict()
            if isinstance(optimizer, optim.Adam):
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_pth = save_dir + f'{optim_name}_' \
                                  f'{model_name}_' \
                                  f'{dset_name}_' \
                                  f'{e+1}epoch_' \
                                  f'{cur_lr:.6f}_' \
                                  f'{train_loss:.3f}loss(train)_' \
                                  f'{val_loss:.3f}loss_' \
                                  f'{val_loss_ap:.3f}loss(ap)_' \
                                  f'{val_loss_ds:.3f}loss(ds)_' \
                                  f'{val_loss_lr:.3f}loss(lr).ckpt'
            torch.save(save_dict, save_pth)

    x_axis = [i for i in range(epoch)]

    plt.figure(0)
    plt.plot(x_axis, train_loss_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_list, 'b-', label='Val')
    plt.title('Loss')
    plt.legend()

    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    graph_pth = graph_dir + f'{model_name}_{date_str}_{time_str}.png'
    plt.savefig(graph_pth)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--weight_decay', type=float, required=False, default=.0005)
    parser.add_argument('--momentum', type=float, required=False, default=.9)
    parser.add_argument('--epoch', type=int, required=False, default=50)
    parser.add_argument('--dset_root', type=str, required=False, default='D://DeepLearningData/KITTI')
    parser.add_argument('--model_ckpt_pth', type=str, required=False, default='')    # Blank if none
    parser.add_argument('--in_size', type=tuple, required=False, default=(256, 512))
    parser.add_argument('--num_classes', type=int, required=False, default=9)    # KITTI objects
    parser.add_argument('--optimizer', type=str, required=False, default='Adam')    # Adam or SGD
    parser.add_argument('--save_dir', type=str, required=False, default='./save/depth/')
    parser.add_argument('--record_dir', type=str, required=False, default='./records/depth/')
    parser.add_argument('--graph_dir', type=str, required=False, default='./graph/depth/')

    args = parser.parse_args()

    torch.set_num_threads(1)
    main(args)