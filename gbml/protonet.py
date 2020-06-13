import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gbml.gbml import GBML
from utils import get_accuracy, apply_grad, mix_grad, grad_to_cos, loss_to_ent

class ProtoNet(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()

        return None

    @torch.enable_grad()
    def inner_loop(self, fmodel, train_inputs, train_targets):

        proto_list = fmodel.get_proto(train_inputs, train_targets)

        return proto_list

    def outer_loop(self, batch, is_train):

        self.network.zero_grad()
        
        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch)

        loss_log = 0
        acc_log = 0
        grad_list = []
        loss_list = []

        fmodel = self.network

        train_inputs = fmodel.extract_feature(train_inputs, mode='stl')
        proto_list = self.inner_loop(fmodel, train_inputs, train_targets)

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