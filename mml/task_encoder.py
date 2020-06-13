import torch
import torch.nn as nn
import torch.nn.functional as F

from net import ConvNet

def dense(in_dim, out_dim, hidden_dim, n_dense):
    layers = [None] * (2*n_dense + 1)
    shapes = [in_dim] + [hidden_dim for i in range(n_dense)] + [out_dim]
    layers[::2] = [nn.Linear(shapes[i],shapes[i+1]) for i in range(len(shapes)-1)]
    layers[1::2] = [nn.ReLU(True) for i in range(n_dense)]
    return nn.Sequential(*layers)

class TaskEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.out_features = args.hidden_channels
        self.hidden_channels = 64
        self.feature_size = args.feature_size
        self.batch_size = args.batch_size
        self.num_way = args.num_way

        self.encoder = ConvNet(args, is_decoder=False)
        self.class_enc = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, self.hidden_channels),
        )
        self.scaler_gen = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, self.out_features*2),
        )
        self.init_params()
        return None
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        return None

    def forward(self, x, y, word_embeddings=None, _lambda=None):
        # Encode inputs
        x = self.encoder(x)
        x = x.reshape(x.size(0),x.size(1),-1)
        x = x.mean(dim=-1)

        # Calculate Proto
        proto = []
        for j in range(self.num_way):
            idx =  (y == j)
            proto.append(x[idx].mean(0))
        proto = torch.stack(proto, dim=0)

        # Convex combination with word embeddings
        if not word_embeddings is None:
            proto = _lambda*proto + (1-_lambda)*word_embeddings

        # Encode proto and Calculate task representation
        proto = self.class_enc(proto)
        task_rep = proto.mean(dim=0,keepdim=True)

        # Get Scaler
        scaler = self.scaler_gen(task_rep).squeeze(dim=0)
        scale, shift = (scaler[:self.out_features].reshape(1,-1,1,1)+1), scaler[self.out_features:].reshape(1,-1,1,1)

        return scale, shift

