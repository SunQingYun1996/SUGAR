# SUGAR
Code for "SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism"
# Overview
- `train.py`: the core of our model, including the structure and the process of training.
- `env.py, QLearning.py`: the code about RL method
- `GCN.py, layers.py, SAGE.py`: including the basic layers we used in the main model.
- `dataset/`: including the dataset. [MUTAG](https://github.com/SunQingYun1996/SUGAR/tree/main/dataset/MUTAG), [DD](), [NCI1](), [NCI109](), [PTC_MR](), [ENZYMES](), [PROTEINS]() (the download link will be provided later).
  - `'RAW/'`: the original data of the dataset
  - `adj.npy`: the biggest Adjacency Matrix built from dataset
  - `graph_label.npy`: the label of every sub_graph
  - `sub_adj.npy`: the Adjacency Matrix of subgraph through sampling
  - `features.npy`: the pre-handled features of each subgraph
# Datasets
- `MUTAG`: The MUTAG dataset consists of 188 chemical compounds divided into two 
classes according to their mutagenic effect on a bacterium. 
- `D&D`: D&D is a dataset of 1178 protein structures (Dobson and Doig, 2003). Each protein is
represented by a graph, in which the nodes are amino acids and two nodes are connected
by an edge if they are less than 6 Angstroms apart. The prediction task is to classify
the protein structures into enzymes and non-enzymes.
- `NCI1`&`NCI109`:NCI1 and NCI109 represent two balanced subsets of datasets of chemical compounds screened
for activity against non-small cell lung cancer and ovarian cancer cell lines respectively
(Wale and Karypis (2006) and http://pubchem.ncbi.nlm.nih.gov).

- `ENZYMES`: ENZYMES is a dataset of protein tertiary structures obtained from (Borgwardt et al., 2005)
consisting of 600 enzymes from the BRENDA enzyme database (Schomburg et al., 2004).
In this case the task is to correctly assign each enzyme to one of the 6 EC top-level
classes.


# Setting
1. mkdir "dataset" & download the dataset into it
2. setting up python env
3. run `python train.py`(all the parameters could be viewed in the `train.py`)
## parameters
```bash
 --dataset DATASET
 --num_info NUM_INFO
 --lr LR (learning_rate)
 --max_pool MAX_POOL
 --momentum MOMENTUM
 --num_epoch NUM_EPOCH
 --batch_size BATCH_SIZE
 --sg_encoder SG_ENCODER(GIN, GCN, GAT, SAGE)
 --MI_loss MI_LOSS
 --start_k START_K
```

# Reference
````
@inproceedings{wang2020gcn,
  title={AM-GCN: Adaptive Multi-channel Graph Convolutional Networks},
  author={Wang, Xiao and Zhu, Meiqi and Bo, Deyu and Cui, Peng and Shi, Chuan and Pei, Jian},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1243--1253},
  year={2020}
}

@nproceedings{sun2021sugar,
  title={SUGAR: Subgraph Neural Network with Reinforcement Pooling and Self-Supervised Mutual Information Mechanism},
  author={Sun, Qingyun and Li, Jianxin and Peng, Hao and Wu, Jia and Ning, Yuanxing and Yu, Phillip S and He, Lifang},
  booktitle={Proceedings of the 2021 World Wide Web Conference},
  year={2021}
}
````
