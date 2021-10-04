# Pileup_GNN
This repository is the implementation in PyTorch for the paper "Semi-supervised Graph Neural Networks for Pileup Per Particle Identification".

**Datasets**: The raw datasets are stored in the `/datasets` folder with the `prepare_dataset.py` to generate graph for each event.
- Fast simulation datasets are the dataset from Pileup mitigation at the Large Hadron Collider
with Graph Neural Networks [paper](https://arxiv.org/pdf/1810.07988.pdf). The dataset can be obtrained from [here]().
- Real simulation dataset is a more realistic setting of pileup simulation, which can be obtained from [here]().
- `/fast_simulation` directory contains the training and testing files on fast simulation dataset.
- `/real_simulation` directory contains the training and testing files on real simulation dataset.

**Dependencies**
- Python ==3.8
- Torch  ==1.7.1
- numpy ==1.20.1
- torch_geometric == 1.6.3

**Training**\
Before start training the models, you should first run `prepare_dataset.py` in `/datasets` to construct the graph for each event of your selected PU level.\
*Fast simulation dataset:* Training can be on both supervised setting and semi-supervised setting. Semi-supervised setting trains on selected charged particles as shown in our paper. Supervised training is trained on all neutral particles which only. 
- Semi-supervised training: in `/fast_simulation` directory, run
```bash
 python train_fastsim_semi.py
 ``` 
- Supervised training: in `/real_simulation` directory, run 
```bash
 python train_fastsim_sup.py
 ``` 

*Real simulation dataset:* Training can only be in semi-supervised setting since there are no labels for neutral particles \
In `/real_simulation` directory, run
```bash
 python train_realsim.py
 ``` 

**Testing**







