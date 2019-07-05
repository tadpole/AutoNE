# AutoNE
The Implementation of "[AutoNE: Hyperparameter Optimization for Massive Network Embedding](https://tadpole.github.io/files/2019_KDD_AutoNE.pdf)"(KDD 2019).

### Requirements
- Python 3
```
$ pip3 install -r requirements.txt
```

### Uasge
The Dataset can be downloaded from [here](https://cloud.tsinghua.edu.cn/f/73d0675acf134f259bf4/?dl=1)

You can change 'dataset', 'method', 'task', 'ms' variables in Makefile to select data and model.

```
dataset :   [BlogCatalog | Wikipedia | pubmed]
method  :   [deepwalk | AROPE | gcn]
task    :   [link_predict | classification]
ms      :   [mle | random_search | b_opt]
```

#### Sampling dataset
```
$ make sample
```

#### Run the model
```
$ make run
```

### Cite
If you find this code useful, please cite our paper:

@inproceedings{tu2019autone,
  title={AutoNE: Hyperparameter Optimization for Massive Network Embedding},
  author={Tu, Ke and Ma, Jianxin and Cui, Peng and Pei, Jian and Zhu, Wenwu},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \&amp; Data Mining},
  year={2019},
  organization={ACM}
}
