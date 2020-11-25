# We refer https://github.com/aerinkim/glove_pretrain

import pickle
import bcolz
import numpy as np

glove_path = '../../data/glove'
words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/840B.300.dat', mode='w')

with open(f'{glove_path}/glove.840B.300d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = '-'.join(line[:-300]).lower()
        # word = line[:len(line)-300]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[-300:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((-1, 300)), rootdir=f'{glove_path}/840B.300.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/840B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/840B.300_idx.pkl', 'wb'))