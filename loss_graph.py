import os
import argparse
import matplotlib.pyplot as plt


def main(args):
    # record_dir = os.path.join(args.dir, args.model_name)
    record_fn_list = os.listdir(args.record_dir)

    record_pth_list = []
    for fn in record_fn_list:
        record_pth_list.append(os.path.join(args.record_dir, fn))

    train_loss_list = []
    val_loss_list = []

    for pth in record_pth_list:
        with open(pth, 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                train_loss, val_loss = list(map(float, l.strip().split(',')[3:5]))
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)

    x_axis = [i for i in range(len(train_loss_list))]
    plt.plot(x_axis, train_loss_list, 'r-', label='Train')
    plt.plot(x_axis, val_loss_list, 'b-', label='Valid')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', required=False, type=str, default='UDNet9')
    parser.add_argument('--record_dir', required=False, type=str, default='./records/UDNet9/2nd/')

    args = parser.parse_args()

    main(args)