import argparse
import torch
from torch_geometric.data import DataLoader
import models_fastsim as models
import utils
import matplotlib
from copy import deepcopy
import os

matplotlib.use("pdf")
import pickle
from timeit import default_timer as timer
from tqdm import tqdm

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
    parser.add_argument('--hybrid_semi', type=int,
                        help='whether use puppi for hybrid algorithm')
    parser.add_argument('--hybrid_sup', type=int,
                        help='whether use puppi for hybrid algorithm')
    parser.add_argument('--pulevel', type=int,
                        help='pileup level for the dataset')
    parser.add_argument('--deltar', type=float,
                        help='deltaR for connecting particles when building the graph')
    parser.add_argument('--comment_semi', type=bool,
                        help='indicate whether to comment out some lines in model initialization')
    parser.add_argument('--comment_sup', type=bool,
                        help='indicate whether to comment out some lines in model initialization')
    parser.set_defaults(model_type='Gated',
                        num_layers=2,
                        batch_size=1,
                        hidden_dim=20,
                        dropout=0,
                        hybrid_semi=False,
                        hybrid_sup=False,
                        pulevel=140,
                        deltar=0.8,
                        )

    return parser.parse_args()


def train(dataset_test, args, batchsize):
    directory_semi= "Gated_PU80_r08_007_2_20_semi_noboost"
    directory_sup = "Gated_PU80_r08_007_2_20_sup_noboost"
    parent_dir = "/home/liu2112/project"
    path_semi = os.path.join(parent_dir, directory_semi)
    path_sup = os.path.join(parent_dir, directory_sup)

    start = timer()

    # generate masks
    dataset_test = generate_mask(dataset_test)
    testing_loader = DataLoader(dataset_test, batch_size=batchsize)

    # testing on the load model
    model_load_semi = models.GNNStack(dataset_test[0].num_feature_actual, args.hidden_dim, 1, args)
    model_load_semi.load_state_dict(torch.load(path_semi + '/best_valid_model.pt'))
    model_load_semi = model_load_semi.to(device)

    model_load_sup = models.GNNStack(dataset_test[0].num_feature_actual, args.hidden_dim, 1, args)
    model_load_sup.load_state_dict(torch.load(path_sup + '/best_valid_model.pt'))
    model_load_sup = model_load_sup.to(device)

    _, _, test_auc_final_semi, _, test_puppi_auc_final_semi, _, label, pred_semi, puppi= \
        test(testing_loader, model_load_semi)

    _, _, test_auc_final_sup, _, test_puppi_auc_final_sup, _, _, pred_sup, _=\
        test(testing_loader, model_load_sup)

    if args.pulevel == 80:
        postfix = 'PU80'
    elif args.pulevel == 140:
        postfix = 'PU140'
    else:
        postfix = 'PU20'
    """
    utils.plot_roc_lowerleft([label, label, label],
                       [pred_semi, pred_sup, puppi],
                       legends=["Semi-supervised", "Fully-supervised", "PUPPI"],
                       postfix=postfix + "combined_testfinal")
    """
    utils.plot_roc([label, label, label],
                             [pred_semi, pred_sup, puppi],
                             legends=["Semi-supervised", "Fully-supervised", "PUPPI"],
                             postfix=postfix + "combined_testfinal")
    utils.plot_roc_logscale([label, label, label],
                             [pred_semi, pred_sup, puppi],
                             legends=["Semi-supervised", "Fully-supervised", "PUPPI"],
                             postfix=postfix + "combined_testfinal")

    print("Semi final test neutral auc " + str(test_auc_final_semi))
    print("Sup final test neutral auc " + str(test_auc_final_sup))
    print("puppi test neutral auc " + str(test_puppi_auc_final_sup))
    end = timer()
    training_time = end - start
    print("testing time " + str(training_time))

def test(loader, model):
    model.eval()

    pred_all = None
    label_all = None
    puppi_all = None
    x_all = None
    test_mask_all = None
    total_loss = 0

    auc_all_puppi = []
    event_num_neu = []

    for data in loader:
        with torch.no_grad():
            data = data.to(device)
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

            test_mask_index = data.num_feature_actual[0].item()
            test_mask = data.x[:, test_mask_index]

            if test_mask_all != None:
                test_mask_all = torch.cat((test_mask_all, test_mask), 0)
            else:
                test_mask_all = test_mask

            label_neu = label[test_mask == 1]
            label_neu = label_neu.cpu().detach().numpy()
            label = label[test_mask == 1]

            puppi_neu = puppi[test_mask == 1]
            puppi_neu = puppi_neu.cpu().detach().numpy()


            pred = pred[test_mask == 1]
            cur_neu_puppi_auc = utils.get_auc(label_neu, puppi_neu)
            auc_all_puppi.append(cur_neu_puppi_auc)

            label = label.type(torch.float)
            label = label.view(-1, 1)
            total_loss += model.loss(pred, label).item() * data.num_graphs

            event_num_neu.append(label.shape[0])

    total_loss /= len(loader.dataset)

    test_mask_all = test_mask_all.cpu().detach().numpy()
    label_all = label_all.cpu().detach().numpy()
    pred_all = pred_all.cpu().detach().numpy()
    puppi_all = puppi_all.cpu().detach().numpy()
    x_all = x_all.cpu().detach().numpy()

    # here actually the masked ones are neutrals
    label_all_chg = label_all[test_mask_all == 1]
    pred_all_chg = pred_all[test_mask_all == 1]
    puppi_all_chg = puppi_all[test_mask_all == 1]
    x_all = x_all[test_mask_all == 1]

    auc_chg = utils.get_auc(label_all_chg, pred_all_chg)
    auc_chg_puppi = utils.get_auc(label_all_chg, puppi_all_chg)
    acc_chg = utils.get_acc(label_all_chg, pred_all_chg)
    acc_chg_puppi = utils.get_acc(label_all_chg, puppi_all_chg)

    return total_loss, acc_chg, auc_chg, acc_chg_puppi, auc_chg_puppi, None, \
           label_all_chg, pred_all_chg, puppi_all_chg

def generate_mask(dataset):
    # mask all neutrals with pt cut to train
    avg_num_neu_LV = 0
    avg_num_neu_PU = 0
    for graph in dataset:
        graph.num_feature_actual = graph.num_features
        Neutral_index = graph.Neutral_index
        Neutral_feature = graph.x[Neutral_index]
        Neutral_index = Neutral_index[torch.where(Neutral_feature[:, 2] > 0.5)[0]]
        training_mask = Neutral_index

        # construct mask vector for training and testing
        mask_training = torch.zeros(graph.num_nodes, 1)
        mask_training[[training_mask.tolist()]] = 1

        x_concat = torch.cat((graph.x, mask_training), 1)
        graph.x = x_concat

        concat_default = torch.cat((graph.x, graph.x[:, 0: -1]), 1)
        graph.x = concat_default

        num_neutral_LV = graph.y[training_mask] == 1
        num_neutral_LV = sum(num_neutral_LV.type(torch.long))
        num_neutral_PU = graph.y[training_mask] == 0
        num_neutral_PU = sum(num_neutral_PU.type(torch.long))
        avg_num_neu_LV += num_neutral_LV
        avg_num_neu_PU += num_neutral_PU
        graph.num_neutral_PU = num_neutral_PU
        graph.num_neutral_LV = num_neutral_LV

    print("avg number of neutral LV " + str(avg_num_neu_LV / len(dataset)))
    print("avg number of neutral PU " + str(avg_num_neu_PU / len(dataset)))
    return dataset


def main():
    args = arg_parse()
    print(args.model_type)

    if args.pulevel == 20:
        if args.deltar == 0.4:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU20/dataset_ggnn_onehot_deltar04_test_3000", "rb") as fp:
                dataset_test = pickle.load(fp)
        else:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU20/dataset_ggnn_onehot_deltar08_test_3000", "rb") as fp:
                dataset_test = pickle.load(fp)

    elif args.pulevel == 80:
        if args.deltar == 0.4:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU80/dataset_ggnn_onehot_deltar04_test_1000", "rb") as fp:
                dataset_test = pickle.load(fp)
        else:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU80/dataset_ggnn_onehot_deltar08_test_1000", "rb") as fp:
                dataset_test = pickle.load(fp)
    else:
        if args.deltar == 0.4:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU140/dataset_ggnn_onehot_deltar04_test_800", "rb") as fp:
                dataset_test = pickle.load(fp)
        else:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU140/dataset_ggnn_onehot_deltar08_test_800", "rb") as fp:
                dataset_test = pickle.load(fp)

    train(dataset_test, args, 1)


if __name__ == '__main__':
    main()
