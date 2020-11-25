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

class ConvNet(nn.Module):

    def __init__(self, args, is_decoder=True):
        super().__init__()

        self.is_decoder = is_decoder
        self.in_channels = args.in_channels
        self.out_features = args.num_way
        self.hidden_channels = args.hidden_channels

        self.encoder = [conv3x3(self.in_channels, self.hidden_channels, True)] +  [conv3x3(self.hidden_channels, self.hidden_channels, i<3) for i in range(args.n_conv-1)]
        self.encoder += [nn.AvgPool2d(5)]
        # self.encoder += [nn.Flatten(), nn.Linear(self.hidden_channels,2)]
        self.encoder = nn.Sequential(*self.encoder)

        if args.alg == 'MAML':
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_channels, self.out_features),
            )
            # self.decoder = nn.Sequential(
            #     nn.Linear(2, self.out_features),
            # )
        elif args.alg == 'AM3_MAML':
            self.decoder = nn.ParameterList([
                nn.Parameter(torch.ones([self.hidden_channels, self.out_features], requires_grad=True)),
                nn.Parameter(torch.zeros([1,self.out_features], requires_grad=True)),
            ])
            # self.decoder = nn.ParameterList([
            #     nn.Parameter(torch.ones([2, self.out_features], requires_grad=True)),
            #     nn.Parameter(torch.zeros([1,self.out_features], requires_grad=True)),
            # ])
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

    def init_decoder(self):
        nn.init.constant_(self.decoder[0], 1.0)
        nn.init.constant_(self.decoder[1], 0.0)

    def forward(self, x, is_decoder=False):
        
        x = self.encoder(x) # (N, 3, 80, 80) -> (N, 640, 10, 10)
        x = x.reshape(x.shape[0], -1)
        if is_decoder:
            x = self.decoder(x) # (N, out_features)
        
        return x

    def forward_global_decoder(self, x):
        x = self.global_decoder(x.reshape(x.shape[0], -1))
        return x

    @torch.no_grad()
    def get_global_label(self, target, reverse_dict):
        target = torch.tensor([self.label2int_dict[reverse_dict[l.item()]] for l in target]).cuda()
        return target

    def init_global_decoder(self, label_dict, n_dense):
        self.label2int_dict = {}
        self.num_global_class = 64
        for i, (k,v) in enumerate(label_dict.items()):
            self.label2int_dict[v] = i
        self.global_decoder = dense(self.hidden_channels*5*5, self.num_global_class, self.hidden_channels, n_dense).cuda()
