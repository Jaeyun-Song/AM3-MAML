# AM3-MAML
MAML with the initialization induced by word embeddings.

Our code is based on https://github.com/sungyubkim/GBML

* [AM3-MAML]

```python
python3 main.py --download False
```

## Results on miniImagenet

* Without pre-trained encoder (Use 64 channels by default. The exceptions are in parentheses)

|                | 5way 1shot          | 5way 1shot (ours) | 5way 5shot          | 5way 5shot (ours) |
| -------------- | ------------------- | ----------------- | ------------------- | ----------------- |
| MAML           | -                   | -                 | 63.11 (64)          | -                 |
| AM3-MAML       | -                   | -                 | -                   | 66.41 (64)        |

## How to run AM3-MAML

1. Download [Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)](https://github.com/stanfordnlp/GloVe) in Glove github. 
2. Extract glove files. 
3. Install Higher and Torchmeta.
4. Set "--download True" if you need to download miniimagenet and run the command above it.

## Related work

[1] [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) \
[2] [Adaptive Cross-Modal Few-Shot Learning](https://arxiv.org/abs/1902.07104) \
[3] [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) \
[4] [Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples](https://arxiv.org/abs/1903.03096) \
[5] [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825)

## Reference for codes

[1] [GBML](https://github.com/sungyubkim/GBML) for the base \
[2] [Torchmeta](https://github.com/tristandeleu/pytorch-meta) for the dataset \
[3] [glove_pretrain](https://github.com/aerinkim/glove_pretrain) for the pretrained glove embeddings \
[4] [SetTransformer](https://github.com/juho-lee/set_transformer) for SetTransformer modules

## Dependencies

* Python >= 3.6
* Pytorch >= 1.2
* [Higher](https://github.com/facebookresearch/higher) 
* [Torchmeta](https://github.com/tristandeleu/pytorch-meta) 

## Acknowledgement

This work was supported by Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government (MSIT) (No.2019-0-01371, Development of brain-inspired AI with human-like intelligence)
