import pickle
import bcolz
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from set_transformer.modules import *

class Glove(nn.Module):

    def __init__(self, args, label_dict):
    # def __init__(self, data_path, hidden_channels):
        super().__init__()

        self.args = args
        data_path = args.data_path + '/glove'
        hidden_channels = args.hidden_channels

        # Load Glove
        vectors = bcolz.open(f'{data_path}/840B.300.dat')[:]
        words = pickle.load(open(f'{data_path}/840B.300_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{data_path}/840B.300_idx.pkl', 'rb'))
        self.glove = {w: vectors[word2idx[w]] for w in words}

        self.transformer = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(True),
            nn.Dropout(args.drop_rate),
            SAB(300, 300, 4, True),
            nn.Linear(300, hidden_channels),
        )

        # self.word_transformer = nn.Sequential(
        #     nn.Linear(300, 300),
        #     nn.ReLU(True),
        #     nn.Dropout(0.9),
        #     nn.Linear(300, hidden_channels),
        #     nn.ReLU(True),
        # )

        # self.lambda_generator = nn.Sequential(
        #     nn.Linear(hidden_channels, 300),
        #     nn.ReLU(True),
        #     nn.Dropout(0.9),
        #     nn.Linear(300, 1),
        #     nn.Sigmoid(),
        # )

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

    def forward(self, x, is_train=True):
        # embedding = [[self.glove[ele] for ele in x_e] for x_e in x]
        embedding = [[self.embedding_dict[ele] for ele in x_e] for x_e in x]
        embedding = torch.tensor(embedding).cuda().type(torch.float32)
        if is_train:
            embedding = embedding + embedding.new_ones(embedding.shape).normal_(std=1)
        x = self.transformer(embedding)
        # x = self.word_transformer(embedding) # (N, 3, 80, 80) -> (N, 64, 5, 5)
        # _lambda = self.lambda_generator(x)
        return x

    def set_eval(self):
        self.transformer.eval()
        # self.word_transformer.eval()
        # self.lambda_generator.eval()

    def set_train(self):
        self.transformer.train()
        # self.word_transformer.train()
        # self.lambda_generator.train()

if __name__ == '__main__':
    
    model = Glove('../../data/glove', 64).cuda()
    x = [torch.tensor(model.glove['lion']).unsqueeze(0), 
        torch.tensor(model.glove['retriever']).unsqueeze(0), 
        torch.tensor(model.glove['school-bus']).unsqueeze(0)]
    for k in model.glove.keys():
        if 'school' in k and 'bus' in k:
            print(k)
    sim1 = (F.normalize(x[0],dim=-1)*F.normalize(x[1],dim=-1)).sum()
    sim2 = (F.normalize(x[0],dim=-1)*F.normalize(x[2],dim=-1)).sum()
    print(sim1)
    print(sim2)
    # print(model(x).shape)