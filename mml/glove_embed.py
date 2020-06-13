import pickle
import bcolz
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Glove(nn.Module):

    def __init__(self, args, label_dict, hidden_channels=None):
    # def __init__(self, data_path, hidden_channels):
        super().__init__()

        self.args = args
        data_path = args.data_path + '/glove'
        if hidden_channels is None:
            hidden_channels = args.hidden_channels * args.feature_size**2
        else:
            hidden_channels = args.hidden_channels

        # Load Glove
        vectors = bcolz.open(f'{data_path}/840B.300.dat')[:]
        words = pickle.load(open(f'{data_path}/840B.300_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{data_path}/840B.300_idx.pkl', 'rb'))
        self.glove = {w: vectors[word2idx[w]] for w in words}

        self.word_transformer = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(300, hidden_channels),
            nn.ReLU(True),
        )

        self.lambda_generator = nn.Sequential(
            nn.Linear(hidden_channels, 300),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(300, 1),
            nn.Sigmoid(),
        )

        self.embedding_dict = {}
        for v in label_dict.values():
            if not (v in words):
                v_ = v.replace('_','-').lower()
                if (v_ in words):
                    self.embedding_dict[v] = self.glove[v_]
                else:
                    v_ = v.replace('_','').lower()
                    if (v_ in words):
                        self.embedding_dict[v] = self.glove[v_]
                    else:
                        v_ = v.lower().split('_')
                        self.embedding_dict[v] = sum([self.glove[e] for e in v_])
            else:
                self.embedding_dict[v] = self.glove[v]   

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
        # embedding = [[self.glove[ele] for ele in x_e] for x_e in x]
        embedding = [[self.embedding_dict[ele] for ele in x_e] for x_e in x]
        embedding = torch.tensor(embedding).cuda().type(torch.float32)
        x = self.word_transformer(embedding) # (N, 3, 80, 80) -> (N, 64, 5, 5)
        _lambda = self.lambda_generator(x)
        return x, _lambda

    def set_eval(self):
        self.word_transformer.eval()
        self.lambda_generator.eval()

    def set_train(self):
        self.word_transformer.train()
        self.lambda_generator.train()

if __name__ == '__main__':
    
    model = Glove('../../data/glove', 64).cuda()
    x = ['cat', 'dog']
    print(model.glove[x])
    print(model(x).shape)