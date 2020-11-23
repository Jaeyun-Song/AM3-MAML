# ResNet Wide Version as in Qiao's Paper
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from utils import euclidean_metric

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        return None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.in_channels = args.in_channels
        self.out_features = args.num_way
        self.batch_size = args.batch_size
        self.num_way = args.num_way

        cfg = [160, 320, 640]
        layers = [3,3,3]
        self.inplanes = iChannels = int(cfg[0]/2)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(iChannels),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, cfg[0], layers[0], stride=2),
            self._make_layer(BasicBlock, cfg[1], layers[1], stride=2),
            self._make_layer(BasicBlock, cfg[2], layers[2], stride=2),
            nn.AvgPool2d(10),
        )
        if args.alg == 'MAML':
            self.decoder = nn.Sequential(
                nn.Linear(cfg[2], self.out_features),
            )
        elif args.alg == 'MMAML':
            self.decoder = nn.ParameterList([
                nn.Parameter(torch.ones([cfg[2], self.num_way], requires_grad=True)),
                nn.Parameter(torch.zeros([1,self.num_way], requires_grad=True)),
            ])
        self.init_params()
        return None

    def init_params(self):
        for k, v in self.named_parameters():
            if ('conv' in k) or ('meta' in k):
                if ('weight' in k):
                    nn.init.kaiming_uniform_(v)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
            elif ('bn' in k):
                if ('weight' in k):
                    nn.init.constant_(v, 1.0)
                elif ('bias' in k):
                    nn.init.constant_(v, 0.0)
        return None

    def init_decoder(self):
        nn.init.constant_(self.decoder[0], 1.0)
        nn.init.constant_(self.decoder[1], 0.0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, is_decoder=False):
        
        x = self.encoder(x) # (N, 3, 80, 80) -> (N, 640, 10, 10)
        x = x.reshape(x.shape[0], -1)
        if is_decoder:
            x = self.decoder(x) # (N, out_features)
        
        return x

    def extract_feature(self, train_inputs, mode='stl'):
        # (M, N, 3, 80, 80) -> (M, N, 64, 5, 5)
        # extract features by meta learned feature extractor
        # Do NOT share batch statistics
        if mode == 'stl':
            input_var = []
            for i in range(len(train_inputs)):
                input_var.append(self(train_inputs[i],is_decoder=False))
            train_inputs = torch.stack(input_var, dim=0)
        # Share batch statistics
        elif mode == 'mtl':
            train_input_shape = train_inputs.shape[:2]
            train_inputs = self(train_inputs.reshape([-1]+list(train_inputs.shape[2:])),is_decoder=False)
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
        for i in range(self.batch_size):
            proto = []
            for j in range(self.num_way):
                idx =  (y[i] == j)
                proto.append(x[i][idx].mean(0))
            proto = torch.stack(proto, dim=0)
            proto_list.append(proto)
        proto_list = torch.stack(proto_list, dim=0)
        return proto_list