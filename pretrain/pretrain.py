import os
import gc
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from torchmeta.datasets import MiniImagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, ClassSplitter

from net import ConvNet, ResNet
from pretrain.dataloader import PretrainDataset, TransformDataset
from utils import get_accuracy, dict2tsv, set_seed, set_gpu, check_dir, PretrainBestTracker

sparsity_log = []

def train(args, net, opt, lr_scheduler, dataloader):
    net.train()
    loss_list = []
    acc_list = []
    sparsity_log.clear()
    with tqdm(dataloader) as pbar:
        for batch_id, batch in enumerate(pbar):
            batch_x, batch_y = batch
            opt.zero_grad()
            pred = net(batch_x.cuda())
            loss = F.cross_entropy(pred, batch_y.cuda())
            loss.backward()
            opt.step()
            if args.lr_sched:
                lr_scheduler.step()
            loss_list.append(loss.item())
            with torch.no_grad():
                acc_list.append(get_accuracy(pred, batch_y.cuda()).item())
            pbar.set_description('loss = {:.4f} || acc={:.4f}'.format(np.mean(loss_list), np.mean(acc_list)))
    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)
    return loss, acc

@torch.no_grad()
def valid(args, net, opt, dataloader):
    net.eval()
    loss_list = []
    acc_list = []
    sparsity_log.clear()
    with tqdm(dataloader) as pbar:
        for batch_id, batch in enumerate(pbar):
            # batch_x, batch_y = batch
            # pred = net(batch_x.cuda())
            # loss = F.cross_entropy(pred, batch_y.cuda())
            # loss_list.append(loss.item())
            # with torch.no_grad():
            #     acc_list.append(get_accuracy(pred, batch_y.cuda()).item())

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.cuda()
            train_targets = train_targets.cuda()

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.cuda()
            test_targets = test_targets.cuda()

            train_inputs = net.extract_feature(train_inputs, mode='stl')
            proto_list = net.get_proto(train_inputs, train_targets)

            test_inputs = net.extract_feature(test_inputs, mode='stl')
            for i, (test_input, test_target) in enumerate(zip(test_inputs, test_targets)):
                test_logit = net.forward_proto(test_input, proto_list[i])
                outer_loss = F.cross_entropy(test_logit, test_target)
                loss_list.append(outer_loss.item())

                with torch.no_grad():
                    acc_list.append(get_accuracy(test_logit, test_target).item())

            pbar.set_description('loss = {:.4f} || acc={:.4f}'.format(np.mean(loss_list), np.mean(acc_list)))
            if batch_id >= 150:
                break

    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)
    return loss, acc

@PretrainBestTracker
def run_epoch(epoch, args, net, opt, lr_scheduler, train_loader, val_loader):
    res = OrderedDict()
    print('Epoch {}'.format(epoch))
    train_loss, train_acc = train(args, net, opt, lr_scheduler, train_loader)
    val_loss, val_acc = valid(args, net, opt, val_loader)
    res['epoch'] = epoch
    res['train_loss'] = train_loss
    res['train_acc'] = train_acc
    res['val_loss'] = val_loss
    res['val_acc'] = val_acc
    gc.collect()
    return res

def sparsity_hook(m, i, o)->None:
    o = o.detach().cpu().numpy()
    sparsity_log.append((np.count_nonzero(o)/np.size(o)))

def main(args):
    # get data loader
    train_transform =transforms.Compose([
                        transforms.RandomCrop(84, padding=8),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225])),
                        ])
    test_transform = transforms.Compose([
                        transforms.CenterCrop(84),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225]))
                        ])
    feature_size = 1

    dataset = PretrainDataset('../data',args.dataset)
    args.feature_size = feature_size
    args.num_way = len(dataset.label)
    # train_set, val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    # train_set = TransformDataset(train_set, train_transform)
    # val_set = TransformDataset(val_set, test_transform)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    # validation is proto
    train_set, val_set = random_split(dataset, [int(1*len(dataset)), len(dataset)-int(1*len(dataset))])
    train_set = TransformDataset(train_set, train_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_dataset = MiniImagenet(args.data_path, num_classes_per_task=10,
                        meta_split='val', 
                        transform=train_transform,
                        target_transform=Categorical(num_classes=10)
                        )
    valid_dataset = ClassSplitter(valid_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=15)
    val_loader = BatchMetaDataLoader(valid_dataset, batch_size=4,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    # get net
    if args.net == 'ConvNet':
        net = ConvNet(args)
    elif args.net == 'ResNet':
        net = ResNet(args, 4, 10)
        args.hidden_channels = 640
    else:
        raise ValueError('Not supported net.')  
    net.train()
    net.cuda()
    # for k,v in net._modules.items():
    #     if k=='encoder':
    #         v.register_forward_hook(sparsity_hook)
    if args.load:
        net.load_state_dict(torch.load(os.path.join(args.folder_path, args.load_path)))
    # get opt
    if args.opt == 'Adam':
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == 'SGD':
        opt = torch.optim.SGD(net.parameters(), lr=args.lr, nesterov=True, momentum=0.9)
    else:
        raise ValueError('Not supported inner opt.')
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, len(train_loader)*args.num_epoch)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[75,90,105], gamma=0.1)
    args.save_path = '_'.join([str(args.num_shot),args.save_path])
    # iter epochs
    for epoch in range(args.num_epoch):
        res, is_best = run_epoch(epoch, args, net, opt, lr_scheduler, train_loader, val_loader)
        dict2tsv(res, os.path.join(args.folder_path, args.log_path))
        if is_best:
            torch.save(net.state_dict(), os.path.join(args.folder_path, args.save_path))
    return None

def parse_args():
    import argparse
    parser = argparse.ArgumentParser('Pretraining for Meta-Learning')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--log_path', type=str, default='result.tsv')
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--load', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='0 = CPU.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epoch', type=int, default=120) 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_shot', type=int, default=1)
    # net settings
    parser.add_argument('--net', type=str, default='ResNet')
    parser.add_argument('--opt', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--n_conv', type=int, default=6)
    parser.add_argument('--n_dense', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size of dense layers')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    args = parser.parse_args()
    args.folder_path = os.path.join(args.result_path, args.dataset, 'pretrain')
    return args

if __name__=='__main__':
    args = parse_args()
    set_seed(args.seed)
    set_gpu(args.device)
    check_dir(args.folder_path)
    main(args)