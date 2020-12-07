import numpy as np
import torch
import torch.nn as nn
import os

from net import ConvNet

class GBML:
    '''
    Gradient-Based Meta-Learning
    '''
    def __init__(self, args):
        self.args = args
        self.batch_size = self.args.batch_size
        return None

    def _init_net(self):
        if self.args.net == 'ConvNet':
            self.network = ConvNet(self.args)
        self.network.train()
        self.network.cuda()
        return None

    def _init_opt(self):
        if self.args.inner_opt == 'SGD':
            self.inner_optimizer = torch.optim.SGD(self.network.decoder.parameters(), lr=self.args.inner_lr)
        elif self.args.inner_opt == 'Adam':
            self.inner_optimizer = torch.optim.Adam(self.network.decoder.parameters(), lr=self.args.inner_lr, betas=(0.0, 0.9))
        else:
            raise ValueError('Not supported inner optimizer.')
        if self.args.outer_opt == 'SGD':
            self.outer_optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.outer_lr, nesterov=True, momentum=0.9)
        elif self.args.outer_opt == 'Adam':
            self.outer_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.outer_lr)
        else:
            raise ValueError('Not supported outer optimizer.')
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=10, gamma=0.5)
        return None

    def unpack_batch(self, batch):
        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.cuda()
        train_targets = train_targets.cuda()

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.cuda()
        test_targets = test_targets.cuda()
        return train_inputs, train_targets, test_inputs, test_targets

    def inner_loop(self):
        raise NotImplementedError

    def outer_loop(self):
        raise NotImplementedError

    def lr_sched(self):
        self.lr_scheduler.step()
        return None

    def load(self):
        path = os.path.join(self.args.result_path, self.args.alg, self.args.load_path)
        self.network.load_state_dict(torch.load(path))

    def load_encoder(self):
        path = os.path.join(self.args.result_path, self.args.alg, self.args.load_path)
        state_dict = torch.load(path)
        new_state_dict = {}
        for (k,v) in state_dict.items():
            if not 'decoder' in k:
                new_state_dict[k[8:]]=v
        self.network.encoder.load_state_dict(new_state_dict)

    def save(self):
        path = os.path.join(self.args.result_path, self.args.alg, self.args.save_path)
        torch.save(self.network.state_dict(), path)

