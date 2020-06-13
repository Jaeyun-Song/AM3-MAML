import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import euclidean_metric
from net.convnet import conv3x3, dense


class ProtoConvNet(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.in_channels = args.in_channels
        self.out_features = args.num_way
        self.hidden_channels = args.hidden_channels

        self.encoder = [conv3x3(self.in_channels, self.hidden_channels, True)] +  [conv3x3(self.hidden_channels, self.hidden_channels, i<3) for i in range(args.n_conv-1)] +[nn.Flatten()]
        self.encoder = nn.Sequential(*self.encoder)
        self.init_params()

        return None
    
    def init_params(self):
        for k, v in self.named_parameters():
            if ('Conv' in k) or ('Linear' in k):
                if ('weight' in k):
                    nn.init.kaiming_uniform_(v)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
            elif ('Batch' in k):
                if ('weight' in k):
                    nn.init.constant_(v, 1.0)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
        return None

    def forward(self, x):

        x = self.encoder(x) # (N, 3, 80, 80) -> (N, 64 * 5 * 5)

        return x

    def extract_feature(self, train_inputs, mode='stl'):
        # (M, N, 3, 80, 80) -> (M, N, 64, 5, 5)
        # extract features by meta learned feature extractor
        # Do NOT share batch statistics
        if mode == 'stl':
            input_var = []
            for i in range(len(train_inputs)):
                input_var.append(self(train_inputs[i]))
            train_inputs = torch.stack(input_var, dim=0)
        # Share batch statistics
        elif mode == 'mtl':
            train_input_shape = train_inputs.shape[:2]
            train_inputs = self(train_inputs.reshape([-1]+list(train_inputs.shape[2:])))
            train_inputs = train_inputs.reshape(list(train_input_shape)+list(train_inputs.shape[1:]))
        else:
            raise ValueError('Not supported mode.')
        
        return train_inputs

    def forward_proto(self, x, proto):
        # Make a prediction based on proto
        pred = euclidean_metric(x, proto)
        return pred

    def get_proto(self, x, y):
        # calculate proto
        # x: (M, N, feature_dim), y: (M, N)
        proto_list = []
        for i in range(self.args.batch_size):
            proto = []
            for j in range(self.args.num_way):
                idx =  (y[i] == j)
                proto.append(x[i][idx].mean(0))
            proto = torch.stack(proto, dim=0)
            proto_list.append(proto)
        proto_list = torch.stack(proto_list, dim=0)
        return proto_list
        