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
sudo docker run -it --gpus=1 -v/PATH_TO_Pileup_GNN:/Workdir -p 8888:8888 -t yongbinfeng/gnntrainingenv:cuda11.3.0-runtime-torch1.12.1-tg2.2.0-ubuntu20.04_v2
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
- In `/datasets`, `creatingGraph.py` is the script to construct graphs for dataset.
- Download the datasets and change the `iname` variable to the location.
- graph is constructed by connecting particles that are less than some threshold of `deltaR`, you can specify the `deltaR` when running the files. The default is 0.4.
- The number of events you want to construct graphs for can be passed as an argument `num_events`
- The starting event can also be specified using argument `start_event`
- `oname` argument helps specify the name you want to save the constructed graphs with
##### ToDOs:
- [ ] Add the parser to the script so that you can specify the arguments when running the script
- [ ] The conversion from awkward arrays to pandas dataframes can probably be avoided. The graph construction can be done directly on the awkward arrays. There might also be better ways to speed up the graph construction.

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


## Note ##
- Don't forget to change the directory of your downloaded raw datasets in prepare dataset files. \
(line 75, 84 in `prepare_dataset_fastsim.py` and line 20 in `prepare_dataset_realsim.py`)
- Also, note when running prepare dataset files for graph construction. The graphs will be saved to the directory that you run the code. If you want to save the graph to another directory, specify the full path plus name using args.name argument.
- Don't forget to change the parent directory in training and testing files. \
(second line of `train` function in `general_test.py`, `test_fastsim_semi.py`, `test_fastsim_sup.py`, `train_fastsim_semi.py` and `train_fastsim_sup.py`)\
(second line of `plot_discriminator` function in `utils.py`)