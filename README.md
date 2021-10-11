# Pileup_GNN
This repository is the implementation in PyTorch for the paper "Semi-supervised Graph Neural Networks for Pileup Per Particle Identification".

## Datasets ##
- Fast simulation datasets are the dataset from Pileup mitigation at the Large Hadron Collider
with Graph Neural Networks [paper](https://arxiv.org/pdf/1810.07988.pdf). The datasets for different pileup conditions can be obtrained from [here](https://zenodo.org/search?page=1&size=20&q=PuppiML).
- Real simulation dataset is a more realistic setting of pileup simulation, which can be obtained from [here]().
- `/fast_simulation` directory contains the training and testing files on fast simulation dataset.
- `/real_simulation` directory contains the training and testing files on real simulation dataset.

## Dependencies ##
- Python ==3.8
- Torch  ==1.7.1
- numpy ==1.20.1
- torch_geometric == 1.6.3

## Setting up requirements ##
For convience, requirements.txt can be used, run as the following
```bash
pip3 install -r requirements.txt
```
However, this doesn't include torch_geometric related packeages. Tutorials about how to install torch_geometric could be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) (Note: please check CUDA version before installation)

## Construct graphs ##
- In `/datasets`, `prepare_dataset_fastsim.py` and `prepare_dataset_realsim.py` are the files to construct graphs for fast simulation and real simulation dataset
- Download the datasets to the `/dataset` directory
- graph is constructed by connecting particles that are less than some threshold of `deltaR`, you can specify the `deltaR` when running the files. The default is 0.8.
- The number of events you want to construct graphs for can be passed as an argument `--num_events`
- The starting event can also be specified using argument `--start_event`
- After downloading the raw files for datasets, specify the correct root in the files.
- For example, to construct graphs for fast simlation dataset with `deltaR` 0.4 with 3000 events starting from event 0. Run
```bash
 python prepare_dataset_fastsim.py --deltaR 0.4 --num_events 3000 --start_event 0
 ```

## Training ##
Before start training the models, you should first run `prepare_dataset.py` in `/datasets` to construct the graphs as instructed in **Construct graphs** section.\
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
 python train_fastsim_semi.py --model_type 'Gated' --num_layers 2 --hidden_dim 20 --pulevel 80
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

## Testing ##
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
 python test_fastsim_semi.py --model_type 'Gated' --num_layers 2 --hidden_dim 20 --pulevel 140
 ``` 
 
## Saved models ##
There are some pretrained models included in `/saved_models` directory. They can be directly loaded for testing without the training phase following the Testing procedure described above.

## Gilbreth Cluster Helpful tips ##

### Install packages ###
For installing packages, [here](https://www.rcac.purdue.edu/knowledge/gilbreth/run/examples/apps/python/packages) includes all kinds of detials. In general, here are some steps:

 ```bash
module load anaconda/5.1.0-py36
conda-env-mod create -p /depot/mylab/apps/mypackages
module load use.own
module load conda-env/mypackages-py3.6.4
```

#### Install with pip ####
```bash
pip install mpi4py
```

#### Install with Conda ####
```bash
conda install opencv
```

### how to create job scripts ###
To submit job to the cluster, first create job scripts using instruction in [here](https://www.rcac.purdue.edu/knowledge/gilbreth/run/slurm/script)

### how to submit jobs ###
Once job script is created, use instructions in [here](https://www.rcac.purdue.edu/knowledge/gilbreth/run/slurm/submit) to submit jobs to the cluster.

### how to monitor job status and outputs ###
After job submission, use instructions in [here](https://www.rcac.purdue.edu/knowledge/gilbreth/run/slurm/status) to monitor job status and [here](https://www.rcac.purdue.edu/knowledge/gilbreth/run/slurm/output) to check job output.








