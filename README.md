# Pileup_GNN
This repository is the implementation in PyTorch for the paper "Semi-supervised Graph Neural Networks for Pileup Per Particle Identification".

## Datasets ##
Full sim datasets based on NanoAOD.
- Real simulation dataset is a more realistic setting of pileup simulation, which can be obtained from [here]().
- `/real_simulation` directory contains the training and testing files on real simulation dataset.

## Setting up requirements ##
One docker environment has been prepared: `yongbinfeng/gnntrainingenv:cuda11.3.0-runtime-torch1.12.1-tg2.2.0-ubuntu20.04_v1`.

To run the environment, do for example:
```
sudo docker run -it --gpus=1 -v/PATH_TO_Pileup_GNN:/Workdir -p 8888:8888 -t yongbinfeng/gnntrainingenv:cuda11.3.0-runtime-torch1.12.1-tg2.2.0-ubuntu20.04_v1
cd /Workdir
```

Then can open the jupyter notebook with
```
jupyter notebook --allow-root --no-browser --port 8888 --ip 0.0.0.0
```

## Note ##
- Don't forget to change the directory of your downloaded raw datasets in prepare dataset files. \
(line 75, 84 in `prepare_dataset_fastsim.py` and line 20 in `prepare_dataset_realsim.py`)
- Also, note when running prepare dataset files for graph construction. The graphs will be saved to the directory that you run the code. If you want to save the graph to another directory, specify the full path plus name using args.name argument.
- Don't forget to change the parent directory in training and testing files. \
(second line of `train` function in `general_test.py`, `test_fastsim_semi.py`, `test_fastsim_sup.py`, `train_fastsim_semi.py` and `train_fastsim_sup.py`)\
(second line of `plot_discriminator` function in `utils.py`)


## Construct graphs ##
- In `/datasets`, `prepare_dataset_fastsim.py` and `prepare_dataset_realsim.py` are the files to construct graphs for fast simulation and real simulation dataset
- Download the datasets to the `/dataset` directory
- graph is constructed by connecting particles that are less than some threshold of `deltaR`, you can specify the `deltaR` when running the files. The default is 0.8.
- The number of events you want to construct graphs for can be passed as an argument `num_events`
- The starting event can also be specified using argument `start_event`
- `name` argument helps specify the name you want to save the constructed graphs with
- For example, to construct graphs for fast simlation dataset with `deltaR` 0.4 with 3000 events starting from event 0. Run
```bash
 python prepare_dataset_fastsim.py --deltaR 0.4 --num_events 3000 --start_event 0 --name "datasets_fastsim_3000_deltar04_start0"
 ```

## Training ##
Before start training the models, you should first run `prepare_dataset.py` in `/datasets` to construct the training and validation graphs as instructed in **Construct graphs** section.\
\
You can specify arguments for training, or it will follow the default sets in the files. The particular arguments that need to be set are `pulevel` to specify the nPU of the training dataset, the `training_path` and `validation_path` to specify the path for the training and validation graphs being constructed in previous step, plus the `save_dir` to specify the directory you want to save the trained model and some training plots.\
\
*Fast simulation dataset:* Training can be on both supervised setting and semi-supervised setting. Semi-supervised setting trains on selected charged particles as shown in our paper. Supervised training is trained on all neutral particles which only. 
- Semi-supervised training: in `/fast_simulation` directory, run
```bash
 python train_fastsim_semi.py --training_path 'your training graph directory' --validation_path 'your validation graph directory' --save_dir 'the dirctory you wish save all the results to'
 ``` 
 Note that, the full path would be the 'parent direcotory' mentioned above concatenate with the --save_dir. 
 For example, if you want to train on PU80 with 2 layers of gated model with 20 dimension. Run 
 ```bash
 python train_fastsim_semi.py --model_type 'Gated' --num_layers 2 --hidden_dim 20 --pulevel 80 --validation_path ... --training_path ... --save_dir ...
 ``` 
- Supervised training: in `/real_simulation` directory, run 
```bash
 python train_fastsim_sup.py --validation_path ... --training_path ... --save_dir ...
 ``` 

*Real simulation dataset:* Training can only be in semi-supervised setting since there are no labels for neutral particles \
In `/real_simulation` directory, run
```bash
 python train_realsim.py
 ``` 

## Testing ##
After training phase, the trained models will be saved and ready for testing. Testing will directly load the models saved during training.\
Specify the `testing_path` in arguments to load the constructed testing graphs and `load_dir` to load the trained model and save testing plots.\
\
Testing can be done on both charged and neutral particles for semi-supervised learning or neutral particles for supervised learning.
- Fast simulation: There are three testing files in `/fast_simulation` directory, `test_fastsim_semi.py` for semi-supervised, `test_fastsim_sup.py` for supervised and `general_test.py` if you want to compare the neutral performance of both supervised and semi-supervised training. Choose one of the three testing scheme. 
```bash
 python test_fastsim_sup.py --testing_path ... --load_dir ...
 python test_fastsim_semi.py --testing_path ... --load_dir ...
 python general_test.py --testing_path ... --load_dir_semi ... --load_dir_sup ...
 ``` 
 The arguments for testing is the same as training. You should specify the arguments based on the model you want to test and the `pulevel` you want to test on.
 For example, the model you are want to test a semi-supervised 2\*20 gated model on nPU=140, then you can run
 ```bash
 python test_fastsim_semi.py --model_type 'Gated' --num_layers 2 --hidden_dim 20 --pulevel 140 --testing_path ... --load_dir ...
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








