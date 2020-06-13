import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

from gbml.gbml import GBML
from mml.task_encoder import TaskEncoder
from utils import get_accuracy, apply_grad, mix_grad, grad_to_cos, loss_to_ent

class MMAML(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self.network.task_encoder = TaskEncoder(args).cuda()
        # self._init_opt()
        return None

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target):

        train_logit = fmodel.decoder(train_input.reshape(train_input.shape[0], -1))
        inner_loss = F.cross_entropy(train_logit, train_target)
        diffopt.step(inner_loss)

        return None

    def outer_loop(self, batch, reverse_dict_list, is_train):

        self.network.zero_grad()
        
        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)

        loss_log = 0
        acc_log = 0
        grad_list = []
        loss_list = []

        for i, (train_input, train_target, test_input, test_target) in enumerate(zip(train_inputs, train_targets, test_inputs, test_targets)):
            with higher.innerloop_ctx(self.network, self.inner_optimizer, track_higher_grads=is_train) as (fmodel, diffopt):

                fmodel(torch.zeros(1,3,84,84).type(torch.float32).cuda())

                # Convert numerical label to word label
                target = [[reverse_dict_list[i][j] for j in range(self.args.num_way)]]

                # Get transformed word embeddings and lambda
                word_proto, _lambda = fmodel.word_embedding(target)

                # Get scale & shift
                scale, shift = fmodel.task_encoder(train_input, train_target, word_proto.squeeze(dim=0), _lambda.squeeze(dim=0))
                # scale, shift = fmodel.task_encoder(train_input, train_target)

                train_input = fmodel(train_input, [scale, shift])
                for step in range(self.args.n_inner):
                    self.inner_loop(fmodel, diffopt, train_input, train_target)

                _test_logit = fmodel(test_input, [scale, shift])
                test_logit = fmodel.decoder(_test_logit.reshape(_test_logit.size(0),-1))
                outer_loss = F.cross_entropy(test_logit, test_target)
                loss_log += outer_loss.item()/self.batch_size

                with torch.no_grad():
                    acc_log += get_accuracy(test_logit, test_target).item()/self.batch_size
            
                if is_train:
                    outer_loss += 1e-2*sum([(_scale**2 + _shift**2).mean() for _scale, _shift in zip(scale, shift)])

                    # global classification
                    global_target = fmodel.get_global_label(test_target, reverse_dict_list[i])
                    global_logit = fmodel.forward_global_decoder(_test_logit.reshape(_test_logit.size(0),-1))
                    global_cls_loss = F.cross_entropy(global_logit, global_target)
                    outer_loss = 0.9*outer_loss + 0.1*global_cls_loss

                    params = fmodel.parameters(time=0)
                    outer_grad = torch.autograd.grad(outer_loss, params)
                    grad_list.append(outer_grad)
                    loss_list.append(outer_loss.item())
        
        self._lambda = _lambda.detach()

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
            self.inner_optimizer = torch.optim.SGD(list(self.network.decoder.parameters()), lr=self.args.inner_lr)
        elif self.args.inner_opt == 'Adam':
            self.inner_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.inner_lr, betas=(0.0, 0.9))
        else:
            raise ValueError('Not supported inner optimizer.')
        if self.args.outer_opt == 'SGD':
            self.outer_optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.outer_lr, nesterov=True, momentum=0.9)
        elif self.args.outer_opt == 'Adam':
            self.outer_optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.outer_lr)
            # self.outer_optimizer = torch.optim.Adam(self.network.word_embedding.parameters(), lr=self.args.outer_lr)
        else:
            raise ValueError('Not supported outer optimizer.')
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.outer_optimizer, \
            milestones=[int(self.args.num_epoch*6/8)], gamma=0.1)
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.outer_optimizer, step_size=10, gamma=0.5)
        return None