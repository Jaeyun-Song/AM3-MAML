import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict
import os

from torchmeta.datasets import MiniImagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, ClassSplitter
from dataset.miniimagenet import LabelMiniImagenet

from gbml import MMAML, MProtoNet
from mml.glove_embed import Glove
from utils import set_seed, set_gpu, check_dir, dict2tsv, BestTracker, get_label_dict, construct_batch

@torch.no_grad()
def valid(args, model, dataloader, label_dict):

    # model.network.word_embedding.set_eval()
    loss_list = []
    acc_list = []

    # collect test keys for miniimagenet
    key_list = [v for (k,v) in label_dict.items()][80:]
    index_dict = {}
    for i, k in enumerate(key_list):
        index_dict[k] = i
    lambda_list = [[] for _ in range(20)]

    with tqdm(dataloader, total=args.num_valid_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            batch, reverse_dict_list = construct_batch(batch, label_dict)
            loss_log, acc_log = model.outer_loop(batch, reverse_dict_list, is_train=False)

            loss_list.append(loss_log)
            acc_list.append(acc_log)
            pbar.set_description('loss = {:.4f} || acc={:.4f}'.format(np.mean(loss_list), np.mean(acc_list)))
            
            # collect lambda
            # for i, b in enumerate(model._lambda.squeeze(dim=-1)):
            #     for j, l in enumerate(b):
            #         lambda_list[index_dict[reverse_dict_list[i][j]]].append(l.item())

            if batch_idx >= args.num_valid_batches:
                break

    # lambda_list = [np.mean(lambda_) for lambda_ in lambda_list]
    # print(lambda_list)
    # print(np.mean(lambda_list), np.std(lambda_list))
    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)

    return loss, acc

def run_epoch(epoch, args, model, test_loader, label_dict):

    res = OrderedDict()
    print('Epoch {}'.format(epoch))
    test_loss, test_acc = valid(args, model, test_loader, label_dict)

    res['epoch'] = epoch
    res['test_loss'] = test_loss
    res['test_acc'] = test_acc
    
    return res

def main(args):

    args.feature_size = 5

    if args.alg=='MMAML':
        model = MMAML(args)
    elif args.alg=='MProtoNet':
        model = MProtoNet(args)
    else:
        raise ValueError('Not implemented Meta-Learning Algorithm')

    print("=====> Load Pretrained GLOVE")
    label_dict = get_label_dict()
    # if args.alg=='MMAML':
    #     model.network.word_embedding = Glove(args, label_dict, args.hidden_channels).cuda()
    # else:
    #     model.network.word_embedding = Glove(args, label_dict).cuda()
    # model._init_opt()

    print("=====> Load Trained Model")
    if args.load:
        model.load()
    elif args.load_encoder:
        model.load_encoder()

    test_dataset = LabelMiniImagenet(args.data_path, num_classes_per_task=args.num_way,
                        meta_split='test', 
                        transform=transforms.Compose([
                        transforms.CenterCrop(84),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225]))
                        ]),
                        # target_transform=Categorical(num_classes=args.num_way)
                        )
    test_dataset = ClassSplitter(test_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    test_loader = BatchMetaDataLoader(test_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    print("=====> Evaluate Model")
    for epoch in range(1,2):

        run_epoch(epoch, args, model, test_loader, label_dict)

    return None

def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    # experimental settings
    parser.add_argument('--seed', type=int, default=2020,
        help='Random seed.')   
    parser.add_argument('--data_path', type=str, default='../data/',
        help='Path of MiniImagenet.')
    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--log_path', type=str, default='result.tsv')
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--load', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_encoder', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_path', type=str, default='best_model.pth')
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='0 = CPU.')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers for data loading (default: 4).')
    # training settings
    parser.add_argument('--num_epoch', type=int, default=400,
        help='Number of epochs for meta train.') 
    parser.add_argument('--batch_size', type=int, default=4,
        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--num_train_batches', type=int, default=250,
        help='Number of batches the model is trained over (default: 250).')
    parser.add_argument('--num_valid_batches', type=int, default=1000,
        help='Number of batches the model is trained over (default: 150).')
    # meta-learning settings
    parser.add_argument('--num_shot', type=int, default=1,
        help='Number of support examples per class (k in "k-shot", default: 1).')
    parser.add_argument('--num_query', type=int, default=15,
        help='Number of query examples per class (k in "k-query", default: 15).')
    parser.add_argument('--num_way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--alg', type=str, default='MAML')
    # algorithm settings
    parser.add_argument('--n_inner', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--inner_opt', type=str, default='SGD')
    parser.add_argument('--outer_lr', type=float, default=1e-3)
    parser.add_argument('--outer_opt', type=str, default='Adam')
    parser.add_argument('--lr_sched', type=lambda x: (str(x).lower() == 'true'), default=False)
    # network settings
    parser.add_argument('--net', type=str, default='ConvNet')
    parser.add_argument('--n_conv', type=int, default=4)
    parser.add_argument('--n_dense', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    set_gpu(args.device)
    check_dir(args)
    main(args)