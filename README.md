# AM3-MAML
MAML with the initialization induced by word embedding

Our code is based on https://github.com/sungyubkim/GBML

* [AM3-MAML]

```python
python3 main_modal.py
```

## Results on miniImagenet

* Without pre-trained encoder (Use 64 channels by default. The exceptions are in parentheses)

|                | 5way 1shot          | 5way 1shot (ours) | 5way 5shot          | 5way 5shot (ours) |
| -------------- | ------------------- | ----------------- | ------------------- | ----------------- |
| AM3-MAML       | -                   | -                 | -                   | 66.41 (64)        |

## Related work

[1] [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) \
[2] [Adaptive Cross-Modal Few-Shot Learning](https://arxiv.org/abs/1902.07104) \
[3] [Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples](https://arxiv.org/abs/1903.03096) \
[4] [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

## Dependencies

* Python >= 3.6
* Pytorch >= 1.2
* [Higher](https://github.com/facebookresearch/higher) 
* [Torchmeta](https://github.com/tristandeleu/pytorch-meta) 

## Acknowledgement

This work was supported by Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government (MSIT) (No.2019-0-01371, Development of brain-inspired AI with human-like intelligence)
