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
\
You can specify arguments for training, or it will follow the default sets in the files. The particular arguments that need to be set are `pulevel` to specify the nPU of the training dataset.\
\
*Fast simulation dataset:* Training can be on both supervised setting and semi-supervised setting. Semi-supervised setting trains on selected charged particles as shown in our paper. Supervised training is trained on all neutral particles which only. 
- Semi-supervised training: in `/fast_simulation` directory, run
```bash
 python train_fastsim_semi.py
 ``` 
 For example, if you want to train on PU80 with 2 layers of gated model with 20 dimension. Run 
 ```bash
 python train_fastsim_semi.py --model_type='Gated' --num_layers=2 --hidden_dim=20 --pulevel=80
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

**Testing**\
After training phase, the trained models will be saved and ready for testing. Testing will directly load the models saved during training.\
\
Testing can be done on both charged and neutral particles for semi-supervised learning or neutral particles for supervised learning.
- Fast simulation: There are three testing files in `/fast_simulation` directory, `test_fastsim_semi.py` for semi-supervised, `test_fastsim_sup.py` for supervised and `general_test.py` if you want to compare the neutral performance of both supervised and semi-supervised training. Choose one of the three testing scheme. 
```bash
 python test_fastsim_sup.py
 python test_fastsim_semi.py
 python general_test.py
 ``` 
 The arguments for testing is the same as training. You should specify the arguments based on the model you want to test and the `pulevel` you want to test on.
 For example, the model you are want to test a semi-supervised 2\*20 gated model on nPU=140, then you can run
 ```bash
 python test_fastsim_semi.py --model_type='Gated' --num_layers=2 --hidden_dim=20 --pulevel=140
 ``` 
 

**Saved models**\
There are some pretrained models included in `/saved_models` directory. They can be directly loaded for testing without the training phase following the Testing procedure described above.










