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
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--hybrid', type=int,
                        help='whether use puppi for hybrid algorithm')
    parser.add_argument('--pulevel', type=int,
                        help='pileup level for the dataset')
    parser.add_argument('--deltar', type=float,
                        help='deltaR for connecting particles when building the graph')

    parser.set_defaults(model_type='Gated',
                        num_layers=2,
                        batch_size=1,
                        hidden_dim=20,
                        dropout=0,
                        opt='adam',
                        #opt_scheduler='step',
                        #opt_decay_step=100,
                        #opt_decay_rate=0.001,
                        weight_decay=0,
                        lr=0.001,
                        hybrid=False,
                        pulevel=20,
                        deltar=0.4)

    return parser.parse_args()


def train(dataset, dataset_validation, args, batchsize):
    directory = "Gated_PU20_r04_001_2_20_semi_noboost_15v1"
    parent_dir = "/home/liu2112/project"
    path = os.path.join(parent_dir, directory)
    isdir = os.path.isdir(path)

    if isdir == False:
        os.mkdir(path)

    start = timer()

    rotate_mask = 5
    if args.pulevel == 20:
        rotate_mask = 8
        num_select_LV = 3
        num_select_PU = 45
    elif args.pulevel == 80:
        num_select_LV = 10
        num_select_PU = 160
    else:
        num_select_LV = 6
        num_select_PU = 282

    generate_mask(dataset, rotate_mask, num_select_LV, num_select_PU)
    generate_mask(dataset_validation, 1, num_select_LV, num_select_PU)

    training_loader = DataLoader(dataset, batch_size=batchsize)
    validation_loader = DataLoader(dataset_validation, batch_size=batchsize)

    model = models.GNNStack(dataset[0].num_feature_actual, args.hidden_dim, 1, args)
    model.to(device)
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    # train
    epochs_train = []
    epochs_valid = []
    loss_graph = []
    loss_graph_train = []
    loss_graph_train_hybrid = []
    loss_graph_valid = []
    auc_graph_train = []
    auc_graph_train_hybrid = []
    auc_graph_valid = []
    auc_graph_neu_train = []
    auc_graph_neu_train_hybrid = []
    auc_graph_neu_valid = []
    train_accuracy = []
    valid_accuracy = []
    train_accuracy_neu = []
    valid_accuracy_neu = []
    auc_graph_train_puppi = []
    auc_graph_valid_puppi = []
    auc_graph_train_puppi_neu = []
    auc_graph_valid_puppi_neu = []
    train_accuracy_puppi = []
    valid_accuracy_puppi = []
    train_accuracy_puppi_neu = []
    valid_accuracy_puppi_neu = []
    train_fig_names = []
    valid_fig_names = []

    count_event = 0
    best_validation_auc = 0
    converge = False
    converge_num_event = 0
    last_steady_event = 0
    lowest_valid_loss = 10

    while converge == False:
        model.train()
        train_mask_all = None

        t = tqdm(total=len(training_loader), colour='green', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        loss_avg = utils.RunningAverage()
        for batch in training_loader:
            count_event += 1
            epochs_train.append(count_event)
            cur_loss = 0
            feature_with_mask = batch.x
            for iter in range(rotate_mask):
                #print("h")
                num_feature = batch.num_feature_actual[0].item()
                batch.x = torch.cat((feature_with_mask[:, 0:num_feature],feature_with_mask[:, (num_feature+iter)].view(-1, 1), feature_with_mask[:, -num_feature:]), 1)
                batch = batch.to(device)

                pred, _ = model.forward(batch)

                label = batch.y
                train_mask = batch.x[:, num_feature]
                if train_mask_all != None:
                    train_mask_all = torch.cat((train_mask_all, train_mask), 0)
                else:
                    train_mask_all = train_mask

                label = label[train_mask == 1]
                label = label.type(torch.float)
                label = label.view(-1, 1)
                pred = pred[train_mask == 1]

                loss = model.loss(pred, label)
                cur_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()

            cur_loss = cur_loss / rotate_mask
            loss_graph.append(cur_loss)
            loss_avg.update(cur_loss)
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

            if count_event % 100 == 0:
                training_loss, training_loss_hybrid, train_acc, train_auc, train_auc_hybrid, \
                train_puppi_acc, train_puppi_auc, \
                train_acc_neu, train_auc_neu, train_auc_neu_hybrid, \
                train_puppi_acc_neu, train_puppi_auc_neu, train_fig_name = test(
                    training_loader, model, 0, count_event)

                valid_loss, valid_loss_hybrid, valid_acc, valid_auc, valid_auc_hybrid,\
                valid_puppi_acc, valid_puppi_auc, \
                valid_acc_neu, valid_auc_neu, valid_auc_neu_hybrid, \
                valid_puppi_acc_neu, valid_puppi_auc_neu, valid_fig_name = test(
                    validation_loader, model, 1, count_event)

                epochs_valid.append(count_event)
                loss_graph_valid.append(valid_loss)
                loss_graph_train.append(training_loss)
                loss_graph_train_hybrid.append(training_loss_hybrid)
                auc_graph_train_puppi.append(train_puppi_auc)
                auc_graph_valid_puppi.append(valid_puppi_auc)
                auc_graph_train.append(train_auc)
                auc_graph_train_hybrid.append(train_auc_hybrid)
                auc_graph_valid.append(valid_auc)
                auc_graph_neu_train.append(train_auc_neu)
                auc_graph_neu_train_hybrid.append(train_auc_neu_hybrid)
                auc_graph_neu_valid.append(valid_auc_neu)
                auc_graph_train_puppi_neu.append(train_puppi_auc_neu)
                auc_graph_valid_puppi_neu.append(valid_puppi_auc_neu)
                train_accuracy.append(train_acc.item())
                valid_accuracy.append(valid_acc.item())
                train_accuracy_neu.append(train_acc_neu.item())
                valid_accuracy_neu.append(valid_acc_neu.item())
                train_accuracy_puppi.append(train_puppi_acc.item())
                valid_accuracy_puppi.append(valid_puppi_acc.item())
                train_accuracy_puppi_neu.append(train_puppi_acc_neu.item())
                valid_accuracy_puppi_neu.append(valid_puppi_acc_neu.item())
                train_fig_names.append(train_fig_name)
                valid_fig_names.append(valid_fig_name)

                if valid_auc_neu > best_validation_auc:
                    best_validation_auc = valid_auc_neu
                    torch.save(model.state_dict(), path + "/best_valid_model.pt")

                if valid_loss >= lowest_valid_loss:
                    print(
                        "valid loss increase at event " + str(count_event) + "with validation loss " + str(valid_loss))
                    if last_steady_event == count_event - 100:
                        converge_num_event += 1
                        if converge_num_event > 30:
                            converge = True
                            break
                        else:
                            last_steady_event = count_event
                    else:
                        converge_num_event = 1
                        last_steady_event = count_event
                    print("converge num event " + str(converge_num_event))
                else:
                    print("lowest valid loss " + str(valid_loss))
                    lowest_valid_loss = valid_loss

                if count_event == 30000:
                    converge = True
                    break

        t.close()

    end = timer()
    training_time = end - start
    print("training time " + str(training_time))

    utils.plot_training(epochs_train, epochs_valid, loss_graph_train,
                        loss_graph, auc_graph_train, train_accuracy_neu, auc_graph_train_puppi,
                        train_accuracy_puppi_neu,
                        loss_graph_valid, auc_graph_valid, valid_accuracy_neu, auc_graph_valid_puppi,
                        valid_accuracy_puppi_neu,
                        auc_graph_neu_train, auc_graph_train_puppi_neu,
                        auc_graph_neu_valid, auc_graph_valid_puppi_neu
                        )
    if args.hybrid == True:
        utils.plot_boost_train(epochs_valid, loss_graph_train, loss_graph_train_hybrid,
                         auc_graph_train_puppi, auc_graph_train_puppi_neu,
                         auc_graph_train, auc_graph_neu_train,
                         auc_graph_train_hybrid, auc_graph_neu_train_hybrid)

def test(loader, model, indicator, epoch):
    if indicator == 0:
        postfix = 'Train'
    elif indicator == 1:
        postfix = 'Validation'
    else:
        postfix = 'Test'

    model.eval()

    pred_all = None
    pred_hybrid_all = None
    label_all = None
    puppi_all = None
    test_mask_all = None
    mask_all_neu = None
    total_loss = 0
    total_loss_hybrid = 0
    count = 0
    for data in loader:
        count += 1
        if count == epoch and indicator == 0:
            break
        with torch.no_grad():
            num_feature = data.num_feature_actual[0].item()
            test_mask = data.x[:, num_feature]

            data.x = torch.cat((data.x[:, 0:num_feature],test_mask.view(-1, 1), data.x[:, -num_feature:]), 1)
            data = data.to(device)
            # max(dim=1) returns values, indices tuple; only need indices
            pred, pred_hybrid = model.forward(data)
            puppi = data.x[:, data.num_feature_actual[0].item() - 1]
            label = data.y

            if pred_all != None:
                pred_all = torch.cat((pred_all, pred), 0)
                pred_hybrid_all = torch.cat((pred_hybrid_all, pred_hybrid), 0)
                puppi_all = torch.cat((puppi_all, puppi), 0)
                label_all = torch.cat((label_all, label), 0)
            else:
                pred_all = pred
                pred_hybrid_all = pred_hybrid
                puppi_all = puppi
                label_all = label

            mask_neu = data.mask_neu[:, 0]

            if test_mask_all != None:
                test_mask_all = torch.cat((test_mask_all, test_mask), 0)
                mask_all_neu = torch.cat((mask_all_neu, mask_neu), 0)
            else:
                test_mask_all = test_mask
                mask_all_neu = mask_neu

            label = label[test_mask == 1]
            pred = pred[test_mask == 1]
            pred_hybrid = pred_hybrid[test_mask == 1]
            label = label.type(torch.float)
            label = label.view(-1, 1)
            total_loss += model.loss(pred, label).item() * data.num_graphs
            total_loss_hybrid += model.loss(pred_hybrid, label).item() * data.num_graphs

    if indicator == 0:
        total_loss /= min(epoch, len(loader.dataset))
        total_loss_hybrid /= min(epoch, len(loader.dataset))
    else:
        total_loss /= len(loader.dataset)
        total_loss_hybrid /= len(loader.dataset)

    test_mask_all = test_mask_all.cpu().detach().numpy()
    mask_all_neu = mask_all_neu.cpu().detach().numpy()
    label_all = label_all.cpu().detach().numpy()
    pred_all = pred_all.cpu().detach().numpy()
    pred_hybrid_all = pred_hybrid_all.cpu().detach().numpy()
    puppi_all = puppi_all.cpu().detach().numpy()

    label_all_chg = label_all[test_mask_all == 1]
    pred_all_chg = pred_all[test_mask_all == 1]
    pred_hybrid_all_chg = pred_hybrid_all[test_mask_all == 1]
    puppi_all_chg = puppi_all[test_mask_all == 1]

    label_all_neu = label_all[mask_all_neu == 1]
    pred_all_neu = pred_all[mask_all_neu == 1]
    pred_hybrid_all_neu = pred_hybrid_all[mask_all_neu == 1]
    puppi_all_neu = puppi_all[mask_all_neu == 1]

    auc_chg = utils.get_auc(label_all_chg, pred_all_chg)
    auc_chg_hybrid = utils.get_auc(label_all_chg, pred_hybrid_all_chg)
    auc_chg_puppi = utils.get_auc(label_all_chg, puppi_all_chg)
    acc_chg = utils.get_acc(label_all_chg, pred_all_chg)
    acc_chg_puppi = utils.get_acc(label_all_chg, puppi_all_chg)

    auc_neu = utils.get_auc(label_all_neu, pred_all_neu)
    auc_neu_hybrid = utils.get_auc(label_all_neu, pred_hybrid_all_neu)
    auc_neu_puppi = utils.get_auc(label_all_neu, puppi_all_neu)
    acc_neu = utils.get_acc(label_all_neu, pred_all_neu)
    acc_neu_puppi = utils.get_acc(label_all_neu, puppi_all_neu)

    utils.plot_roc([label_all_chg, label_all_chg, label_all_neu, label_all_neu],
                   [pred_all_chg, puppi_all_chg, pred_all_neu, puppi_all_neu],
                   legends=["prediction Chg", "PUPPI Chg", "prediction Neu", "PUPPI Neu"],
                   postfix=postfix + "_test")

    fig_name_prediction = utils.plot_discriminator(epoch,
                                                   [pred_all_chg[label_all_chg == 1], pred_all_chg[label_all_chg == 0],
                                                    pred_all_neu[label_all_neu == 1],
                                                    pred_all_neu[label_all_neu == 0]],
                                                   legends=['LV Chg', 'PU Chg', 'LV Neu', 'PU Neu'],
                                                   postfix=postfix + "_prediction", label='Prediction')
    fig_name_puppi = utils.plot_discriminator(epoch,
                                              [puppi_all_chg[label_all_chg == 1], puppi_all_chg[label_all_chg == 0],
                                               puppi_all_neu[label_all_neu == 1],
                                               puppi_all_neu[label_all_neu == 0]],
                                              legends=['LV Chg', 'PU Chg', 'LV Neu', 'PU Neu'],
                                              postfix=postfix + "_puppi", label='PUPPI Weight')

    return total_loss, total_loss_hybrid, acc_chg, auc_chg, auc_chg_hybrid, acc_chg_puppi, auc_chg_puppi, \
           acc_neu, auc_neu, auc_neu_hybrid, acc_neu_puppi, auc_neu_puppi, fig_name_prediction


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
            #gen_index_LV = random.sample(range(LV_index.shape[0]), num_select_LV)
            selected_LV_train = LV_index[(num*num_select_LV):((num+1)*num_select_LV)]

            #gen_index_PU = random.sample(range(PU_index.shape[0]), num_select_PU)
            selected_PU_train = PU_index[(num*num_select_PU):((num+1)*num_select_PU)]

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
                                    torch.zeros(graph.num_nodes, 1), torch.ones(graph.num_nodes, 1)), 1)

        puppiWeight_default_one_hot_training = puppiWeight_default_one_hot_training.type(torch.float32)

        # -3 is for one hot encoding of fromLV; -1 is for final puppiweight; want to have eta, phi, pt as original
        default_data_training = torch.cat(
            (original_feature[:, 0:(graph.num_feature_actual - 3 - 1)], puppiWeight_default_one_hot_training,
             original_feature[:, -1].view(-1, 1)), 1)

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
    print(args.model_type)

    # resample the training but not reconstruct the graph
    if args.pulevel == 20:
        if args.deltar == 0.4:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU20/dataset_ggnn_onehot_deltar04_train_9000", "rb") as fp:
                dataset = pickle.load(fp)
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU20/dataset_ggnn_onehot_deltar04_validation_3000", "rb") as fp:
                dataset_validation = pickle.load(fp)
        else:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU20/dataset_ggnn_onehot_deltar08_train_9000", "rb") as fp:
                dataset = pickle.load(fp)
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU20/dataset_ggnn_onehot_deltar08_validation_3000", "rb") as fp:
                dataset_validation = pickle.load(fp)

    elif args.pulevel == 80:
        if args.deltar == 0.4:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU80/dataset_ggnn_onehot_deltar04_train_3000", "rb") as fp:
                dataset = pickle.load(fp)
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU80/dataset_ggnn_onehot_deltar04_validation_1000",
                      "rb") as fp:
                dataset_validation = pickle.load(fp)
        else:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU80/dataset_ggnn_onehot_train_3000", "rb") as fp:
                dataset = pickle.load(fp)
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU80/dataset_ggnn_onehot_validation_1000",
                      "rb") as fp:
                dataset_validation = pickle.load(fp)
    else:
        if args.deltar == 0.4:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU140/dataset_ggnn_onehot_deltar04_train_3000", "rb") as fp:
                dataset = pickle.load(fp)
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU140/dataset_ggnn_onehot_deltar04_validation_800",
                      "rb") as fp:
                dataset_validation = pickle.load(fp)
        else:
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU140/dataset_ggnn_onehot_train_1000", "rb") as fp:
                dataset = pickle.load(fp)
            with open("/scratch/gilbreth/liu2112/ggnn_graphs_PU140/dataset_ggnn_onehot_validation_400",
                      "rb") as fp:
                dataset_validation = pickle.load(fp)

    generate_neu_mask(dataset)
    generate_neu_mask(dataset_validation)
    train(dataset, dataset_validation, args, 1)


if __name__ == '__main__':
    main()
