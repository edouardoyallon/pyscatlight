import argparse
import os
import time
import numpy as np
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from pyscatlight import Scatlight as Scattering

import torchvision.models as models
import models as mymodels

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + \
              sorted(name for name in mymodels.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(mymodels.__dict__[name]))

parser = argparse.ArgumentParser(description='Benchmarks')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: scat_resnet_big)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--without_scat', dest='dont_measure_scat', action='store_true',
                    help='measure scat in timing')
parser.add_argument('--bottom_only', dest='bottom_only', action='store_true',
                    help='bottom')
parser.add_argument('--num_trials', default=100, type=int, help='trials')
#### ADDED by Edouard
parser.add_argument('--J', '--scale_scattering', default=3, type=int, metavar='N',
                    help='scale to select to compute the order 1 scattering')
parser.add_argument('--bottleneck_width', default='[128,256]', type=str, help='size of the bottleneck')
parser.add_argument('--bottleneck_depth', default='[3,3]', type=str, help='size of the bottleneck')
parser.add_argument('--bottleneck_conv1x1', default=0, type=int, help='number of 1x1')


def main():
    global args
    args = parser.parse_args()
    if args.bottom_only:
        N=1024
    else:
        N=224
    use_scat = False
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        if args.bottom_only:
            model = nn.Sequential(model.conv1, model.bn1,model.relu,
                            model.maxpool,model.layer1,model.layer2,model.layer3)


    else:
        print("=> creating model '{}'".format(args.arch))
        args.bottleneck_width = json.loads(args.bottleneck_width)
        args.bottleneck_depth = json.loads(args.bottleneck_depth)
        global scat
        scat = Scattering(M=N, N=N, J=args.J, pre_pad=False).cuda()
        model = mymodels.__dict__[args.arch](N, args.J,
                                             width = args.bottleneck_width,
                                             depth= args.bottleneck_depth,
                                             conv1x1=args.bottleneck_conv1x1)
        if args.bottom_only:
            lis = nn.ModuleList([model.conv1, model.bn1, model.relu])
            if model.block_of_1x1:
                lis += list(model.conv1block)

            model = nn.Sequential(*lis, model.layers[0], model.layers[1])

        use_scat = True

    print(model)
    model.cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print('Number of parameters: %d and batch size %d'%(params,args.batch_size))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    batch_time = AverageMeter()
    trial_time = []

    # Evaluate model in inference mode
    model.eval()
    for i in range(args.num_trials):
        data = torch.cuda.FloatTensor(args.batch_size,3,N,N).normal_()
        if use_scat:
            data.cuda()
        else:
            data = torch.autograd.Variable(data).cuda()

        start = time.perf_counter()
        if use_scat:
            data = torch.autograd.Variable(scat(data))
        if args.dont_measure_scat:
            torch.cuda.synchronize()
            start = time.perf_counter()
        model(data)
        torch.cuda.synchronize()
        final_t = time.perf_counter() - start
        if i == 0: # Lets ignore the first run
            continue
        batch_time.update(final_t)
        trial_time.append(final_t)
        if i % args.print_freq == 0:
            print('Time val {batch_time.val:.3f} Avg: ({batch_time.avg:.3f}) Median: {median:.3f} Std: {std: .3f}\t'.format(
                   batch_time=batch_time,median=np.median(np.array(trial_time)), std=np.std(np.array(trial_time))))

    print('Final Inference Avg: ({batch_time.avg:.3f}) Median: {median:.3f} Std: {std: .3f}\t'.format(
        batch_time=batch_time, median=np.median(np.array(trial_time)), std=np.std(np.array(trial_time))))

    batch_time = AverageMeter()
    trial_time = []
    model.train()
    dist = torch.distributions.Categorical(torch.ones(1000) / 1000.)
    for i in range(args.num_trials):
        data = torch.cuda.FloatTensor(args.batch_size, 3, N, N).normal_()
        if use_scat:
            data.cuda()
        else:
            data = torch.autograd.Variable(data).cuda()
        lis = [dist.sample() for i in range(args.batch_size)]
        target_var = torch.autograd.Variable(torch.cat(lis,0)).cuda()

        start = time.perf_counter()
        if use_scat:
            data = torch.autograd.Variable(scat(data))
        if args.dont_measure_scat:
            torch.cuda.synchronize()
            start = time.perf_counter()
        output = model(data)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        if i == 0: # Lets ignore the first run
            continue
        final_t = time.perf_counter() - start
        batch_time.update(final_t)
        trial_time.append(final_t)
        if i % args.print_freq == 0:
            print(
                'Time val {batch_time.val:.3f} Avg: ({batch_time.avg:.3f}) Median: {median:.3f} Std: {std: .3f}\t'.format(
                    batch_time=batch_time, median=np.median(np.array(trial_time)), std=np.std(np.array(trial_time))))

    print('Final Train Avg: ({batch_time.avg:.3f}) Median: {median:.3f} Std: {std: .3f}\t'.format(
        batch_time=batch_time, median=np.median(np.array(trial_time)), std=np.std(np.array(trial_time))))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




if __name__ == '__main__':
    main()
