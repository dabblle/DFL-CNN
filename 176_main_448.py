from model.DFL import DFL_VGG16
from utils.util import *
from utils.transform import *
from train import *
from validate import *
from utils.init import *
import sys
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.MyImageFolderWithPaths import *
from drawrect import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataroot', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--result', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batchsize_per_gpu', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-testbatch', '--test_batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--init_type', default='xavier', type=str,
                    metavar='INIT', help='init net')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='momentum', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.000005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=4, type=int,
                    help='GPU nums to use.')
parser.add_argument('--log_train_dir', default='log_train', type=str,
                    help='log for train')
parser.add_argument('--log_test_dir', default='log_test', type=str,
                    help='log for test')
parser.add_argument('--nclass', default=583, type=int,
                    help='num of classes')
parser.add_argument('--eval_epoch', default=2, type=int,
                    help='every eval_epoch we will evaluate')
parser.add_argument('--vis_epoch', default=2, type=int,
                    help='every vis_epoch we will evaluate')
parser.add_argument('--save_epoch', default=2, type=int,
                    help='every save_epoch we will evaluate')
parser.add_argument('--w', default=448, type=int,
                    help='transform, seen as align')
parser.add_argument('--h', default=448, type=int,
                    help='transform, seen as align')

best_prec1 = 0


def main():
    print('DFL-CNN <==> Part1 : prepare for parameters <==> Begin')
    global args, best_prec1
    args = parser.parse_args()
    print('DFL-CNN <==> Part1 : prepare for parameters <==> Done')

    print('DFL-CNN <==> Part2 : Load Network  <==> Begin')
    model = DFL_VGG16(k=10, nclass=176)
    if args.gpu is not None:
        model = nn.DataParallel(model, device_ids=range(args.gpu))
        model = model.cuda()
        cudnn.benchmark = True
    if args.init_type is not None:
        try:
            init_weights(model, init_type=args.init_type)
        except:
            sys.exit('DFL-CNN <==> Part2 : Load Network  <==> Init_weights error!')
    ####################
    weight = np.array([0.844414894, 2.834821429, 1.725543478, 1.322916667, 1.725543478, 1.700892857,
                       0.739518634, 0.629960317, 0.773133117, 1.190625, 1.700892857, 3.501838235,
                       3.052884615, 1.368534483, 1.451981707, 1.831730769, 2.204861111, 1.133928571,
                       0.592350746, 0.262252203, 1.123231132, 0.308452073, 1.280241935, 1.009004237,
                       1.725543478, 1.308379121, 0.265172606, 1.777052239, 1.469907407, 1.00052521,
                       1.803977273, 0.470602767, 0.960181452, 1.67693662, 1.608952703, 0.280807783,
                       0.9921875, 0.466911765, 1.112733645, 2.903963415, 2.768895349, 0.295440447,
                       0.265764509, 2.289663462, 2.38125, 1.434487952, 1.984375, 0.580792683, 1.630993151,
                       1.831730769, 1.860351563, 1.803977273, 2.768895349, 1.951844262, 2.126116071, 1.831730769,
                       1.920362903, 2.768895349, 1.777052239, 3.217905405, 2.334558824, 2.088815789, 0.519923581,
                       3.133223684, 1.951844262, 2.289663462, 1.133928571, 1.507120253, 1.984375, 2.334558824,
                       1.48828125, 1.984375, 1.5875, 1.831730769, 1.092316514, 1.469907407, 1.26662234, 1.5875,
                       0.661458333, 2.088815789, 1.725543478, 1.725543478, 0.376780063, 1.384447674, 1.630993151,
                       2.9765625, 0.607461735, 0.888526119, 2.645833333, 2.334558824, 1.777052239, 0.519923581,
                       0.875459559, 0.268158784, 0.278183411, 1.017628205, 1.009004237, 0.548675115, 0.264583333,
                       1.860351563, 1.777052239, 0.65418956, 0.712949102, 1.984375, 2.088815789, 0.365222393,
                       1.803977273, 0.564277251, 1.630993151, 1.889880952, 1.507120253, 1.507120253, 1.803977273,
                       2.429846939, 3.96875, 1.67693662, 1.630993151, 2.164772727, 2.052801724, 1.167279412,
                       1.803977273, 0.9525, 1.831730769, 1.700892857, 1.352982955, 1.253289474, 2.164772727,
                       2.164772727, 1.352982955, 2.645833333, 1.831730769, 1.630993151, 1.5875, 3.217905405,
                       0.804476351, 2.705965909, 1.725543478, 1.630993151, 2.429846939, 1.777052239, 0.922965116,
                       2.204861111, 1.123231132, 0.79375, 0.320060484, 0.403601695, 0.543664384, 0.269372172,
                       2.588315217, 2.289663462, 1.322916667, 1.280241935, 0.259395425, 1.951844262, 0.620117188,
                       0.252251059, 1.653645833, 0.519923581, 0.616904145, 4.252232143, 2.164772727, 1.920362903,
                       1.725543478, 2.052801724, 1.700892857, 0.413411458, 1.400735294, 1.035326087, 1.526442308,
                       0.881944444, 2.126116071, 2.246462264, 1.984375, 1.984375, 1.889880952, 1.860351563])

    weight = torch.Tensor(weight)
    weight = weight.cuda()
    ####################
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('DFL-CNN <==> Part2 : Load Network  <==> Continue from {} epoch {}'.format(args.resume,
                                                                                             checkpoint['epoch']))
        else:
            print('DFL-CNN <==> Part2 : Load Network  <==> Failed')
    print('DFL-CNN <==> Part2 : Load Network  <==> Done')

    print('DFL-CNN <==> Part3 : Load Dataset  <==> Begin')
    dataroot = os.path.abspath(args.dataroot)
    traindir = os.path.join(dataroot, 'train')
    testdir = os.path.join(dataroot, 'test')

    # ImageFolder to process img
    transform_train = get_transform_for_train()
    transform_test = get_transform_for_test()
    transform_test_simple = get_transform_for_test_simple()

    train_dataset = ImageFolderWithPaths(traindir, transform=transform_train)
    test_dataset = ImageFolderWithPaths(testdir, transform=transform_test)
    test_dataset_simple = ImageFolderWithPaths(testdir, transform=transform_test_simple)

    # A list for target to classname
    index2classlist = train_dataset.index2classlist()

    # data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.gpu * args.train_batchsize_per_gpu, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader_simple = torch.utils.data.DataLoader(
        test_dataset_simple, batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    print('DFL-CNN <==> Part3 : Load Dataset  <==> Done')

    print('DFL-CNN <==> Part4 : Train and Test  <==> Begin')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch, gamma=0.1)

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if epoch % args.eval_epoch == 0:
            prec1 = validate_simple(args, test_loader_simple, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'prec1': prec1,
            }, is_best)

            # do a test for visualization
        if epoch % args.vis_epoch == 0 and epoch != 0:
            draw_patch(epoch, model, index2classlist, args)


if __name__ == '__main__':
    main()
