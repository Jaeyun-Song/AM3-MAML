import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, use_maxpool=False):
    block = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    ]
    if use_maxpool:
        block.append(nn.MaxPool2d(2))
    return nn.Sequential(*block)

def dense(in_dim, out_dim, hidden_dim, n_dense):
    layers = [None] * (2*n_dense + 1)
    shapes = [in_dim] + [hidden_dim for i in range(n_dense)] + [out_dim]
    layers[::2] = [nn.Linear(shapes[i],shapes[i+1]) for i in range(len(shapes)-1)]
    layers[1::2] = [nn.ReLU(True) for i in range(n_dense)]
    return nn.Sequential(*layers)

class FiLMConvNet(nn.Module):

    def __init__(self, args, is_decoder=True):
        super().__init__()

        self.is_decoder = is_decoder
        self.in_channels = args.in_channels
        self.out_features = args.num_way
        self.hidden_channels = args.hidden_channels
        self.n_conv = args.n_conv
        self.batch_size = args.batch_size

        # self.encoder = [conv3x3(self.in_channels, self.hidden_channels, True)] +  [conv3x3(self.hidden_channels, self.hidden_channels, i<3) for i in range(args.n_conv-1)]
        self.encoder = [
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.hidden_channels),
            )]
        self.encoder += [
            nn.Sequential(
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.hidden_channels),
            ) for _ in range(args.n_conv-1)]
        self.relu = nn.ReLU(True)
        self.maxpool2d = nn.MaxPool2d(2)
        self.encoder = nn.ModuleList(self.encoder)
        self.encoder2 = nn.Sequential(
                nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.hidden_channels),
                nn.ReLU(True),
                )

        if self.is_decoder:
            self.decoder = dense(self.hidden_channels*5*5, self.out_features, args.hidden_dim, args.n_dense)
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

    def forward(self, x, scaler=None, mode='decoder'):

        for i in range(self.n_conv):
            x = self.encoder[i](x)
            if not scaler is None:
                x = x * scaler[0][i] + scaler[1][i]
            x = self.relu(x)
            if i < 4:
                x = self.maxpool2d(x)
        if not scaler is None:
            x_before = x.clone()
            x = self.encoder2(x)
            x = 0.5*x + 0.5*x_before
        
        if self.is_decoder and mode == 'decoder' and scaler is None:
            x = self.decoder(x.reshape(x.shape[0], -1)) # (N, out_features)

        return x

    def forward_global_decoder(self, x):
        x = self.global_decoder(x.reshape(x.shape[0], -1))
        return x

    @torch.no_grad()
    def get_global_label(self, target, reverse_dict):
        target = torch.tensor([[self.label2int_dict[reverse_dict[i][l.item()]] for l in target[i]] for i in range(self.batch_size)]).cuda()
        target = target.reshape(-1)
        return target

    def init_global_decoder(self, label_dict, hidden_dim, n_dense):
        self.label2int_dict = {}
        self.num_global_class = 64
        for i, (k,v) in enumerate(label_dict.items()):
            self.label2int_dict[v] = i
        self.global_decoder = dense(self.hidden_channels*5*5, self.num_global_class, hidden_dim, n_dense).cuda()
