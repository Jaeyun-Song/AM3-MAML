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
| AM3-MAML       | -                   | -                 | 66.41 (64)          | -                 |

## Dependencies

* Python >= 3.6
* Pytorch >= 1.2
* [Higher](https://github.com/facebookresearch/higher) 
* [Torchmeta](https://github.com/tristandeleu/pytorch-meta) 

