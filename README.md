# Pileup_GNN

**Datasets**
The raw datasets are stored in the `/datasets` folder with the `prepare_dataset.py` to generate graph for each event.
- Fast simulation datasets are the dataset from Pileup mitigation at the Large Hadron Collider
with Graph Neural Networks [paper](https://arxiv.org/pdf/1810.07988.pdf)
- Real simulation dataset is a more realistic setting of pileup simulation
- `/fast_simulation` directory contains the training and testing files on fast simulation dataset.
- `/real_simulation` directory contains the training and testing files on real simulation dataset. 

**Dependencies**
- Python ==3.8
- Torch  ==1.7.1
- numpy ==1.20.1
- torch_geometric == 1.6.3

**Implementation**


