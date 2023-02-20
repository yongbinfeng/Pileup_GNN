import argparse
import torch
from torch_geometric.data import DataLoader
import models_fastsim as models
import utils
import matplotlib
from copy import deepcopy
import os

matplotlib.use("pdf")
import numpy as np
import random
import pickle
from timeit import default_timer as timer
from tqdm import tqdm
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--pulevel', type=int,
                        help='pileup level for the dataset')
    parser.add_argument('--deltar', type=float,
                        help='deltaR for connecting particles when building the graph')
    parser.add_argument('--testing_path', type=str, required=True,
                        help='path for the testing graphs')
    parser.add_argument('--load_dir', type=str, required=True,
                        help='directory to load the trained model and save the testing plots')

    parser.set_defaults(model_type='Gated',
                        num_layers=2,
                        batch_size=1,
                        hidden_dim=20,
                        dropout=0,
                        pulevel=20,
                        deltar=0.4
                        )

    return parser.parse_args()


def train(dataset_test, args, batchsize):
    directory = args.load_dir
    #parent_dir = "/home/liu2112/project"
    parent_dir = "/home/gpaspala/new_Pileup_GNN/Pileup_GNN/fast_simulation/test"
    path = os.path.join(parent_dir, directory)
    isdir = os.path.isdir(path)

    if isdir == False:
        os.mkdir(path)

    rotate_mask = 5
    if args.pulevel == 20:
        rotate_mask = 8
        num_select_LV = 3
        num_select_PU = 27
    elif args.pulevel == 80:
        num_select_LV = 10
        num_select_PU = 160
    else:
        num_select_LV = 6
        num_select_PU = 282

    start = timer()

    generate_mask(dataset_test, rotate_mask, num_select_LV, num_select_PU)
    testing_loader = DataLoader(dataset_test, batch_size=batchsize)

    # testing on the load model
    model_load = models.GNNStack(dataset_test[0].num_feature_actual, args.hidden_dim, 1, args)
    model_load.load_state_dict(torch.load(path + '/best_valid_model.pt'))
    model_load = model_load.to(device)

    for name, param in model_load.named_parameters():
        if param.requires_grad:
            if name == "beta_one":
                print(param.data)

    test_loss_final, test_acc_final, test_auc_final, test_puppi_acc_final, test_puppi_auc_final, test_acc_neu_final, test_auc_neu_final, test_puppi_acc_neu_final, test_puppi_auc_neu_final, test_fig_name_final = test(
        testing_loader,
        model_load, args,
        "final")

    print("final test neutral auc " + str(test_auc_neu_final))
    print("puppi test neutral auc " + str(test_puppi_auc_neu_final))
    end = timer()
    training_time = end - start
    print("testing time " + str(training_time))


def test(loader, model, args, epoch):
    if args.pulevel == 80:
        postfix = 'PU80'
    elif args.pulevel == 140:
        postfix = 'PU140'
    else:
        postfix = 'PU20'

    model.eval()

    pred_all = None
    label_all = None
    puppi_all = None
    x_all = None
    test_mask_all = None
    mask_all_neu = None
    total_loss = 0
    count = 0

    auc_all_puppi = []
    event_num_neu = []
    event_num_chg = []

    for data in loader:
        count += 1
        if count == epoch and indicator == 0:
            break
        with torch.no_grad():
            num_feature = data.num_feature_actual[0].item()
            # num_mask = data.num_mask[0]
            # we mask 5 times for charged particles, when evaluation we can just sum to get all the mask
            test_mask = data.x[:, num_feature]

            data.x = torch.cat((data.x[:, 0:num_feature], test_mask.view(-1, 1), data.x[:, -num_feature:]), 1)
            data = data.to(device)
            # max(dim=1) returns values, indices tuple; only need indices
            _, pred = model.forward(data)
            puppi = data.x[:, data.num_feature_actual[0].item() - 1]
            label = data.y

            if pred_all != None:
                pred_all = torch.cat((pred_all, pred), 0)
                puppi_all = torch.cat((puppi_all, puppi), 0)
                label_all = torch.cat((label_all, label), 0)
                x_all = torch.cat((x_all, data.x), 0)
            else:
                pred_all = pred
                puppi_all = puppi
                label_all = label
                x_all = data.x

            # test_mask = data.x[:, test_mask_index]
            mask_neu = data.mask_neu[:, 0]

            if test_mask_all != None:
                test_mask_all = torch.cat((test_mask_all, test_mask), 0)
                mask_all_neu = torch.cat((mask_all_neu, mask_neu), 0)
            else:
                test_mask_all = test_mask
                mask_all_neu = mask_neu

            label_neu = label[mask_neu == 1].cpu().detach().numpy()
            puppi_neu = puppi[mask_neu == 1].cpu().detach().numpy()
            #cur_neu_puppi_auc = utils.get_auc(label_neu, puppi_neu)
            #auc_all_puppi.append(cur_neu_puppi_auc)

            label = label[test_mask == 1]
            pred = pred[test_mask == 1]
            label = label.type(torch.float)
            label = label.view(-1, 1)
            total_loss += model.loss(pred, label).item() * data.num_graphs

            event_num_chg.append(label.size()[0])
            event_num_neu.append(label_neu.shape[0])

    total_loss /= len(loader.dataset)

    test_mask_all = test_mask_all.cpu().detach().numpy()
    mask_all_neu = mask_all_neu.cpu().detach().numpy()
    label_all = label_all.cpu().detach().numpy()
    pred_all = pred_all.cpu().detach().numpy()
    puppi_all = puppi_all.cpu().detach().numpy()
    x_all = x_all.cpu().detach().numpy()

    label_all_chg = label_all[test_mask_all == 1]
    pred_all_chg = pred_all[test_mask_all == 1]
    puppi_all_chg = puppi_all[test_mask_all == 1]
    x_all_chg = x_all[test_mask_all == 1]

    label_all_neu = label_all[mask_all_neu == 1]
    pred_all_neu = pred_all[mask_all_neu == 1]
    puppi_all_neu = puppi_all[mask_all_neu == 1]
    x_all_neu = x_all[mask_all_neu == 1]

    auc_chg = utils.get_auc(label_all_chg, pred_all_chg)
    auc_chg_puppi = utils.get_auc(label_all_chg, puppi_all_chg)
    acc_chg = utils.get_acc(label_all_chg, pred_all_chg)
    acc_chg_puppi = utils.get_acc(label_all_chg, puppi_all_chg)

    auc_neu = utils.get_auc(label_all_neu, pred_all_neu)
    auc_neu_puppi = utils.get_auc(label_all_neu, puppi_all_neu)
    acc_neu = utils.get_acc(label_all_neu, pred_all_neu)
    acc_neu_puppi = utils.get_acc(label_all_neu, puppi_all_neu)


    utils.plot_roc([label_all_chg, label_all_neu],
                   [pred_all_chg, pred_all_neu],
                   legends=["prediction Chg", "prediction Neu"],
                   postfix=postfix + "_testfinal", dir_name = args.load_dir)

    utils.plot_roc_logscale([label_all_chg, label_all_neu],
                            [pred_all_chg, pred_all_neu],
                            legends=["prediction Chg", "prediction Neu"],
                            postfix=postfix + "_testfinal", dir_name = args.load_dir)
   


    utils.plot_roc_lowerleft([label_all_chg, label_all_chg],
                             [pred_all_chg, puppi_all_chg],
                             legends=["prediction Chg", "prediction Neu"],
                             postfix=postfix + "_testfinal", dir_name = args.load_dir)



    fig_name_prediction = utils.plot_discriminator(epoch,
                                                   [pred_all_chg[label_all_chg == 1], pred_all_chg[label_all_chg == 0],
                                                    pred_all_neu[label_all_neu == 1],
                                                    pred_all_neu[label_all_neu == 0]],
                                                    legends=['LV Chg', 'PU Chg', 'LV Neu', 'PU Neu'],
                                                   postfix=postfix + "_prediction", label='Prediction', dir_name = args.load_dir)

    return total_loss, acc_chg, auc_chg, acc_chg_puppi, auc_chg_puppi, acc_neu, auc_neu, acc_neu_puppi, auc_neu_puppi, fig_name_prediction


def generate_mask(dataset, num_mask, num_select_LV, num_select_PU):
    # how many LV and PU to sample
    for graph in dataset:
        LV_index = graph.LV_index
        PU_index = graph.PU_index
        np.random.shuffle(LV_index)
        np.random.shuffle(PU_index)
        original_feature = graph.x[:, 0:graph.num_feature_actual]

        mask_training = torch.zeros(graph.num_nodes, num_mask)
        for num in range(num_mask):
            if LV_index.shape[0] < num_select_LV or PU_index.shape[0] < num_select_PU:
                num_select_LV = min(LV_index.shape[0], num_select_LV)
                num_select_PU = min(PU_index.shape[0], num_select_PU)

            # generate the index for LV and PU samples for training mask
            # gen_index_LV = random.sample(range(LV_index.shape[0]), num_select_LV)
            selected_LV_train = LV_index[(num * num_select_LV):((num + 1) * num_select_LV)]

            # gen_index_PU = random.sample(range(PU_index.shape[0]), num_select_PU)
            selected_PU_train = PU_index[(num * num_select_PU):((num + 1) * num_select_PU)]

            training_mask = np.concatenate((selected_LV_train, selected_PU_train), axis=None)
            # print(training_mask)

            # construct mask vector for training and testing
            mask_training_cur = torch.zeros(graph.num_nodes)
            mask_training_cur[[training_mask.tolist()]] = 1
            mask_training[:, num] = mask_training_cur

        x_concat = torch.cat((original_feature, mask_training), 1)
        graph.x = x_concat

        # mask the puppiWeight as default Neutral(here puppiweight is actually fromLV in ggnn dataset)
        puppiWeight_default_one_hot_training = torch.cat((torch.zeros(graph.num_nodes, 1),
                                                          torch.zeros(graph.num_nodes, 1),
                                                          torch.ones(graph.num_nodes, 1)), 1)

        puppiWeight_default_one_hot_training = puppiWeight_default_one_hot_training.type(torch.float32)


        pdgId_one_hot_training = torch.cat((torch.zeros(graph.num_nodes, 1),
                                                         torch.zeros(graph.num_nodes, 1),
                                                         torch.ones(graph.num_nodes, 1)), 1)       
        pdgId_one_hot_training = pdgId_one_hot_training.type(torch.float32)

        pf_dz_training_test=torch.clone(original_feature[:,6:7])
        #print ("pf_dz_training_test: ", pf_dz_training_test)
        #print ("pf_dz_training_test: ", pf_dz_training_test.shape)
        #pf_dz_training_test[[training_mask.tolist()],0]=0
        pf_dz_training_test = torch.zeros(graph.num_nodes, 1)        


        default_data_training = torch.cat(
             (original_feature[:, 0:(graph.num_feature_actual - 7)],pdgId_one_hot_training, pf_dz_training_test ,puppiWeight_default_one_hot_training), 1)



        concat_default = torch.cat((graph.x, default_data_training), 1)
        graph.x = concat_default
        graph.num_mask = num_mask


def generate_neu_mask(dataset):
    # all neutrals with pt cuts are masked for evaluation
    for graph in dataset:
        nparticles = graph.num_nodes
        graph.num_feature_actual = graph.num_features
        Neutral_index = graph.Neutral_index
        Neutral_feature = graph.x[Neutral_index]
        Neutral_index = Neutral_index[torch.where(Neutral_feature[:, 2] > 0.5)[0]]

        mask_neu = torch.zeros(nparticles, 1)
        mask_neu[Neutral_index, 0] = 1
        graph.mask_neu = mask_neu

    return dataset


def main():
    args = arg_parse()
    print("model type: ", args.model_type)

    with open(args.testing_path, "rb") as fp:
        dataset_test = pickle.load(fp)

    generate_neu_mask(dataset_test)

    train(dataset_test, args, 1)


if __name__ == '__main__':
    main()
