import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gbml.gbml import GBML
from utils import get_accuracy, apply_grad, mix_grad, grad_to_cos, loss_to_ent

class MProtoNet(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        # self._init_opt()

        return None

    @torch.enable_grad()
    def inner_loop(self, fmodel, train_inputs, train_targets, reverse_dict_list):
        # Get Proto from inputs
        proto_list = fmodel.get_proto(train_inputs, train_targets)

        # Convert numerical label to word label
        target = [[reverse_dict_list[i][j] for j in range(self.args.num_way)] for i in range(self.args.batch_size)]

        # Get transformed word embeddings and lambda
        word_proto, _lambda = fmodel.word_embedding(target)
        self._lambda = _lambda.detach()

        # Convex Combination of proto and word embedding
        proto_list = _lambda*proto_list + (1-_lambda)*word_proto

        return proto_list

    def outer_loop(self, batch, reverse_dict_list, is_train):

        self.network.zero_grad()
        
        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)

        loss_log = 0
        acc_log = 0
        grad_list = []
        loss_list = []

        fmodel = self.network

        train_inputs = fmodel.extract_feature(train_inputs, mode='stl')
        proto_list = self.inner_loop(fmodel, train_inputs, train_targets, reverse_dict_list)

        test_inputs = fmodel.extract_feature(test_inputs, mode='stl')
        for (i, test_input, test_target) in zip(np.arange(self.args.batch_size), test_inputs, test_targets):
            test_logit = fmodel.forward_proto(test_input, proto_list[i])
            outer_loss = F.cross_entropy(test_logit, test_target)
            loss_log += outer_loss.item() / self.batch_size

            with torch.no_grad():
                acc_log += get_accuracy(test_logit, test_target).item() / self.batch_size
    
        if is_train:
            outer_grad = torch.autograd.grad(outer_loss, fmodel.parameters())
            grad_list.append(outer_grad)
            loss_list.append(outer_loss.item())

        if is_train:
            weight = torch.ones(len(grad_list))
            weight = weight / torch.sum(weight)
            grad = mix_grad(grad_list, weight)
            grad_log = apply_grad(self.network, grad)
            self.outer_optimizer.step()
            
            return loss_log, acc_log, grad_log
        else:
            return loss_log, acc_log

    def _init_opt(self):
        if self.args.inner_opt == 'SGD':
            self.inner_optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.inner_lr)
            # self.inner_optimizer = torch.optim.SGD(self.network.word_embedding.parameters(), lr=self.args.inner_lr)
            # self.inner_optimizer = torch.optim.SGD(list(self.network.encoder.parameters())+list(self.network.decoder.parameters()), lr=self.args.inner_lr)
        elif self.args.inner_opt == 'Adam':
            self.inner_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.inner_lr, betas=(0.0, 0.9))
        else:
            raise ValueError('Not supported inner optimizer.')
        if self.args.outer_opt == 'SGD':
            self.outer_optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.outer_lr, nesterov=True, momentum=0.9)
        elif self.args.outer_opt == 'Adam':
            # self.outer_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.outer_lr)
            self.outer_optimizer = torch.optim.Adam(self.network.word_embedding.parameters(), lr=self.args.outer_lr)
        else:
            raise ValueError('Not supported outer optimizer.')
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=10, gamma=0.5)
        return None