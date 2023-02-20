# Graph PUPPI
This repository collects the code for the study of apply semi-supervised GNN for pileup mitigation, on the full Geant-based simulation samples.

The original code is based on [this repository](https://github.com/sallylsk/Pileup_GNN), with the paper [here](https://arxiv.org/abs/2203.15823). The studies in the paper and code are done on Delphes dataset.

## Datasets ##
Full sim datasets based on customized CMS NanoAODs with some additional variables to help compare with PUPPI performances and understand the differences. We start with $Z\to\nu\nu$ HT datasets.

## Setting up requirements ##
One docker environment has been prepared: `yongbinfeng/gnntrainingenv:cuda11.3.0-runtime-torch1.12.1-tg2.2.0-ubuntu20.04_v3`, which should include all the modules and packages to run the full pipeline.

To run the environment, firstly clone the code
```
git clone git@github.com:yongbinfeng/Pileup_GNN.git
```
With docker, do for example:
```
sudo docker run -it --gpus=1 -v/PATH_TO_Pileup_GNN:/Workdir -p 8888:8888 -t yongbinfeng/gnntrainingenv:cuda11.3.0-runtime-torch1.12.1-tg2.2.0-ubuntu20.04_v3
cd /Workdir
```
(Port number 8888 is only needed for the Jupyter notebook.)

For sites supporting singularity, you can also run the environment with singularity:
```
singularity pull gnntrainingenv.sif docker://yongbinfeng/gnntrainingenv:cuda11.3.0-runtime-torch1.12.1-tg2.2.0-ubuntu20.04_v3
singularity run --nv -B /PATH_TO_Pileup_GNN:/Workdir gnntrainingenv.sif
```
Image only needs to be pulled once (`singularity pull`).

Then can open the jupyter notebook with
```
jupyter notebook --allow-root --no-browser --port 8888 --ip 0.0.0.0
```

### SetUp
The code contains three major ingredients: graph generation, training, and performance testing.

### Build graphs ##
- In `/graphconstruction`, `creatingGraph.py` is the script to construct graphs for dataset.
- Download the datasets and change the `iname` variable to the location.
- graph is constructed by connecting particles that are less than some threshold of `deltaR`, you can specify the `deltaR` when running the files. The default is 0.4.
- The number of events you want to construct graphs for can be passed as an argument `num_events`
- The starting event can also be specified using argument `start_event`
- `oname` argument helps specify the name you want to save the constructed graphs with
##### ToDOs:
- [ ] Add the parser to the script so that you can specify the arguments when running the script
- [ ] The conversion from awkward arrays to pandas dataframes can probably be avoided. The graph construction can be done directly on the awkward arrays. There might also be better ways to speed up the graph construction.

### Training ###
Graphs need to be constructed before running the training code.

You can specify arguments for training, or it will follow the default sets in the files. The particular arguments that need to be set are `pulevel` to specify the nPU of the training dataset, the `training_path` and `validation_path` to specify the path for the training and validation graphs being constructed in previous step, plus the `save_dir` to specify the directory you want to save the trained model and some training plots.\
\
*Fast simulation dataset:* Training can be on both supervised setting and semi-supervised setting. Semi-supervised setting trains on selected charged particles as shown in our paper. Supervised training is trained on all neutral particles which only. 
- Semi-supervised training: run
```bash
 python training/train_semi.py --training_path 'your training graph directory' --validation_path 'your validation graph directory' --save_dir 'the dirctory you wish save all the results to'
 ``` 
 Note that, the full path would be the 'parent direcotory' mentioned above concatenate with the --save_dir. 
 For example, if you want to train on PU80 with 2 layers of gated model with 20 dimension. Run 
 ```bash
 python training/train_semi.py --model_type 'Gated' --num_layers 2 --hidden_dim 20 --pulevel 80 --validation_path ... --training_path ... --save_dir ...
 ``` 
- Supervised training: to be updated

### Testing ###
Use the `testing/test_physics_metrics.py` for now.
