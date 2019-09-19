from __future__ import print_function
import argparse
import os
import random
import shutil
import time
import warnings
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import sys
sys.path.append('./vision')
import torchvision.transforms as transforms
from torchvision.transforms.img_proc import *
import torchvision.datasets as datasets
import torchvision.models as models

from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data','-d', metavar='DATA', default='cub',
                    help='dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--save_path', '-s', metavar='SAVE', default='',
                    help='saving path')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr1', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr2', default=0.001, type=float,
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
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--is_fix', dest='is_fix', action='store_true',
                    help='is_fix.')
parser.add_argument('--is_att', dest='is_att', action='store_true',
                    help='is_att.')
                       
''' loss params '''
parser.add_argument('--w_cls', dest='w_cls', default=1, type=float,
                    help='loss weight for L_cls.')


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ''' random seed '''
    if args.seed is not None:
        random.seed(args.seed)
    else:
        args.seed = random.randint(1, 10000)
        
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    print('==> random seed:',args.seed)
    

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    ''' data load info '''
    data_info = h5py.File(os.path.join('./data',args.data,'data_info.h5'), 'r')
    img_path = str(data_info['img_path'][...])
    nc = data_info['all_att'][...].shape[0]
    sf_size = data_info['all_att'][...].shape[1]
    semantic_data = {'seen_class':data_info['seen_class'][...],
                     'unseen_class': data_info['unseen_class'][...],
                     'all_class':np.arange(nc),
                     'all_att': data_info['all_att'][...]}
    
    # create model
    params = {'num_classes':nc,'is_fix':args.is_fix, 'sf_size':sf_size, 'is_att':args.is_att}
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        best_prec1=0
        model,criterion = models.__dict__[args.arch](pretrained=True,params=params)
    else:
        print("=> creating model '{}'".format(args.arch))
        model,criterion = models.__dict__[args.arch](params=params)
    print("=> is the backbone fixed: '{}'".format(args.is_fix))

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda(args.gpu)


    ''' optionally resume from a checkpoint'''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            if(best_prec1==0):
                best_prec1 = checkpoint['best_prec1']
            print('=> pretrained acc {:.4F}'.format(best_prec1))
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join('./data',args.data,'train.list')
    valdir1 = os.path.join('./data',args.data,'test_seen.list')
    valdir2 = os.path.join('./data',args.data,'test_unseen.list')

    train_transforms, val_transforms = preprocess_strategy(args.data)

    train_dataset = datasets.ImageFolder(img_path,traindir,train_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader1 = torch.utils.data.DataLoader(
        datasets.ImageFolder(img_path,valdir1, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    val_loader2 = torch.utils.data.DataLoader(
        datasets.ImageFolder(img_path,valdir2, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    # evaluate on validation set
    prec1 = validate(train_loader, val_loader1, val_loader2, semantic_data, model, criterion)
        

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()


def validate(train_loader, val_loader1, val_loader2, semantic_data, model, criterion):

    ''' load semantic data'''
    seen_c = semantic_data['seen_class']
    unseen_c = semantic_data['unseen_class']
    all_c = semantic_data['all_class']
    label_s =  torch.from_numpy(seen_c).cuda(args.gpu).long()
    label_t =  torch.from_numpy(unseen_c).cuda(args.gpu).long()
    all_sf = torch.from_numpy(semantic_data['all_att']).cuda(args.gpu,non_blocking=True)
   
    h5_semantic_file = h5py.File('./data.h5', 'w') 

    sf = semantic_data['all_att']
    h5_semantic_file.create_dataset('sf', sf.shape,dtype=np.float32)
    h5_semantic_file['sf'][...] = sf


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        
        for i, (input, target) in enumerate(train_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            # inference
            logits,feat  = model(input,all_sf)
            v_feat = feat[0]
            a_feat = feat[1]
			
            # evaluation
            if(i==0):
                gt_train = target.cpu().numpy()
                feat_train = a_feat
            else:
                gt_train = np.hstack([gt_train,target.cpu().numpy()])
                feat_train = np.vstack([feat_train,a_feat])

            print(i)
            
            
        for i, (input, target) in enumerate(val_loader1):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
           # inference
            logits,feat  = model(input,all_sf)
            v_feat = feat[0]
            a_feat = feat[1]
			
            # evaluation
            if(i==0):
                gt_s = target.cpu().numpy()
                feat_s = a_feat
            else:
                gt_s = np.hstack([gt_s,target.cpu().numpy()])
                feat_s = np.vstack([feat_s,a_feat])

            print(i)
            

            print(i)
        for i, (input, target) in enumerate(val_loader2):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
                
            # inference
            logits,feat  = model(input,all_sf)
            v_feat = feat[0]
            a_feat = feat[1]
			
            # evaluation
            if(i==0):
                gt_t = target.cpu().numpy()
                feat_t = a_feat
            else:
                gt_t = np.hstack([gt_t,target.cpu().numpy()])
                feat_t = np.vstack([feat_t,a_feat])


            print(i)

 
        h5_semantic_file.create_dataset('gt_t',gt_t.shape,dtype=np.int32)
        h5_semantic_file.create_dataset('unseen_c', unseen_c.shape,dtype=np.int32)
        h5_semantic_file.create_dataset('gt_s',gt_s.shape,dtype=np.int32)
        h5_semantic_file.create_dataset('seen_c', seen_c.shape,dtype=np.int32)
        h5_semantic_file.create_dataset('feat_s', feat_s.shape,dtype=np.float32)
        h5_semantic_file.create_dataset('feat_t', feat_t.shape,dtype=np.float32)
        h5_semantic_file.create_dataset('gt_train',gt_train.shape,dtype=np.int32)
        h5_semantic_file.create_dataset('feat_train', feat_train.shape,dtype=np.float32)   


        h5_semantic_file['gt_t'][...] = gt_t
        h5_semantic_file['unseen_c'][...] = unseen_c
        h5_semantic_file['gt_s'][...] = gt_s
        h5_semantic_file['seen_c'][...] = seen_c
        h5_semantic_file['feat_s'][...] = feat_s
        h5_semantic_file['feat_t'][...] = feat_t
        h5_semantic_file['gt_train'][...] = gt_train
        h5_semantic_file['feat_train'][...] = feat_train                 


        h5_semantic_file.close()      



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, optimizer1, optimizer2 , epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr1 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    lr = args.lr1 * (0.1 ** (epoch // 30))
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
        
    lr = args.lr2 * (0.1 ** (epoch // 30))
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
