import argparse
import torch
from torch_geometric.data import DataLoader
import models_ggnn_oldpipe as models
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
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--pulevel', type=int,
                        help='pileup level for the dataset')
    parser.add_argument('--deltar', type=float,
                        help='deltaR for connecting particles when building the graph')
    parser.add_argument('--training_path', type=str,
                        help='path for the training graphs')
    parser.add_argument('--validation_path', type=str,
                        help='path for the validation graphs')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='directory to save the trained model and plots')

    parser.set_defaults(model_type='Gated',
                        num_layers=2,
                        batch_size=1,
                        hidden_dim=20,
                        dropout=0,
                        opt='adam',
                        weight_decay=0,
                        lr=0.007,
                        pulevel=80,
                        deltar=0.4
                        )

    return parser.parse_args()


def train(dataset, dataset_validation, args, batchsize):
    directory = args.save_dir
    parent_dir = "/home/liu2112/project"
    path = os.path.join(parent_dir, directory)
    isdir = os.path.isdir(path)

    if isdir == False:
        os.mkdir(path)

    start = timer()

    # generate masks
    dataset = generate_mask(dataset)
    dataset_validation = generate_mask(dataset_validation)

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
    epochs_test = []
    loss_graph_test = []
    loss_graph_valid = []
    auc_graph_train = []
    auc_graph_test = []
    train_accuracy = []
    test_accuracy = []
    auc_graph_train_puppi = []
    auc_graph_test_puppi = []
    train_accuracy_puppi = []
    test_accuracy_puppi = []
    train_fig_names = []
    test_fig_names = []
    auc_graph_valid = []
    auc_graph_valid_puppi = []
    valid_accuracy = []
    valid_accuracy_puppi = []
    valid_fig_names = []

    count_event = 0
    best_validation_auc = 0
    converge = False
    converge_num_event = 0
    last_steady_event = 0
    lowest_valid_loss = 10
    while converge == False:
        print("start training")
        model.train()
        train_mask_all = None

        t = tqdm(total=len(training_loader), colour='green', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        loss_avg = utils.RunningAverage()
        for batch in training_loader:
            # to determine if the model converges; if so, then stop the training

            count_event += 1
            # print("this is event " + str(count_event))
            epochs_train.append(count_event)
            batch = batch.to(device)
            _, pred = model.forward(batch)
            label = batch.y

            num_feature = batch.num_feature_actual[0].item()
            train_mask = batch.x[:, num_feature]
            if train_mask_all != None:
                train_mask_all = torch.cat((train_mask_all, train_mask), 0)
            else:
                train_mask_all = train_mask

            label = label[train_mask == 1]
            label = label.type(torch.float)
            label = label.view(-1, 1)
            pred = pred[train_mask == 1]
            # pred = torch.transpose(pred, 0, 1)

            loss = model.loss(pred, label)
            cur_loss = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_graph.append(cur_loss)

            loss_avg.update(cur_loss)
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

            # test on the training and validation dataset
            if count_event % 100 == 0:
                training_loss, train_acc, train_auc, train_puppi_acc, train_puppi_auc, train_fig_name = test(
                    training_loader,
                    model, 0,
                    count_event, args)
                valid_loss, valid_acc, valid_auc, valid_puppi_acc, valid_puppi_auc, valid_fig_name = test(
                    validation_loader, model, 1,
                    count_event, args)

                # epochs_train.append(count_event)
                epochs_valid.append(count_event)
                loss_graph_valid.append(valid_loss)
                # avg_training_loss = sum(loss_graph[-100:])
                loss_graph_train.append(training_loss)
                auc_graph_train_puppi.append(train_puppi_auc)
                auc_graph_valid_puppi.append(valid_puppi_auc)
                auc_graph_train.append(train_auc)
                auc_graph_valid.append(valid_auc)
                train_accuracy.append(train_acc.item())
                valid_accuracy.append(valid_acc.item())
                train_accuracy_puppi.append(train_puppi_acc.item())
                valid_accuracy_puppi.append(valid_puppi_acc.item())
                train_fig_names.append(train_fig_name)
                valid_fig_names.append(valid_fig_name)

                # update the best model and test on the testing dataset
                if valid_auc > best_validation_auc:
                    best_validation_auc = valid_auc
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

    # call plot_training_fullsim is just because we only evaluate performance on masked neutral ones
    utils.plot_training_fullsim(epochs_train, epochs_valid, loss_graph_train,
                                loss_graph, auc_graph_train, train_accuracy, auc_graph_train_puppi,
                                train_accuracy_puppi,
                                loss_graph_valid, auc_graph_valid, valid_accuracy, auc_graph_valid_puppi,
                                valid_accuracy_puppi, args.save_dir)


def test(loader, model, indicator, epoch, args):
    if indicator == 0:
        postfix = 'Train'
    elif indicator == 1:
        postfix = 'Validation'
    else:
        postfix = 'Test'

    model.eval()

    pred_all = None
    label_all = None
    puppi_all = None
    test_mask_all = None
    total_loss = 0
    counter = 0
    for data in loader:
        counter += 1
        if counter == epoch and indicator == 0:
            break
        with torch.no_grad():
            data = data.to(device)
            # max(dim=1) returns values, indices tuple; only need indices
            _, pred = model.forward(data)
            puppi = data.x[:, data.num_feature_actual[0].item() - 1]
            label = data.y

            if pred_all != None:
                pred_all = torch.cat((pred_all, pred), 0)
                puppi_all = torch.cat((puppi_all, puppi), 0)
                label_all = torch.cat((label_all, label), 0)
            else:
                pred_all = pred
                puppi_all = puppi
                label_all = label

            test_mask_index = data.num_feature_actual[0].item()
            test_mask = data.x[:, test_mask_index]

            if test_mask_all != None:
                test_mask_all = torch.cat((test_mask_all, test_mask), 0)
            else:
                test_mask_all = test_mask

            label = label[test_mask == 1]
            pred = pred[test_mask == 1]
            label = label.type(torch.float)
            label = label.view(-1, 1)
            total_loss += model.loss(pred, label).item() * data.num_graphs

    if indicator == 0:
        total_loss /= min(epoch, len(loader.dataset))
    else:
        total_loss /= len(loader.dataset)

    test_mask_all = test_mask_all.cpu().detach().numpy()
    label_all = label_all.cpu().detach().numpy()
    pred_all = pred_all.cpu().detach().numpy()
    puppi_all = puppi_all.cpu().detach().numpy()

    # here actually the masked ones are neutrals
    label_all_chg = label_all[test_mask_all == 1]
    pred_all_chg = pred_all[test_mask_all == 1]
    puppi_all_chg = puppi_all[test_mask_all == 1]

    auc_chg = utils.get_auc(label_all_chg, pred_all_chg)
    auc_chg_puppi = utils.get_auc(label_all_chg, puppi_all_chg)
    acc_chg = utils.get_acc(label_all_chg, pred_all_chg)
    acc_chg_puppi = utils.get_acc(label_all_chg, puppi_all_chg)

    utils.plot_roc([label_all_chg, label_all_chg],
                   [pred_all_chg, puppi_all_chg],
                   legends=["prediction Chg", "PUPPI Chg", "prediction Neu", "PUPPI Neu"],
                   postfix=postfix + "_test", dir_name=args.save_dir)
    fig_name_prediction = utils.plot_discriminator(epoch,
                                                   [pred_all_chg[label_all_chg == 1], pred_all_chg[label_all_chg == 0]],
                                                   legends=['LV Neutral', 'PU Neutral'],
                                                   postfix=postfix + "_prediction", label='Prediction', dir_name=args.save_dir)
    fig_name_puppi = utils.plot_discriminator(epoch,
                                              [puppi_all_chg[label_all_chg == 1], puppi_all_chg[label_all_chg == 0]],
                                              legends=['LV Neutral', 'PU Neutral'],
                                              postfix=postfix + "_puppi", label='PUPPI Weight', dir_name=args.save_dir)

    return total_loss, acc_chg, auc_chg, acc_chg_puppi, auc_chg_puppi, fig_name_prediction


def generate_mask(dataset):
    # mask all neutrals with pt cut to train
    avg_num_neu_LV = 0
    avg_num_neu_PU = 0
    avg_num_chg_LV = 0
    avg_num_chg_PU = 0
    avg_num_nodes = 0
    avg_num_edges = 0
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
        avg_num_chg_LV += graph.LV_index.shape[0]
        avg_num_chg_PU += graph.PU_index.shape[0]
        avg_num_nodes += graph.num_nodes
        avg_num_edges += graph.num_edges
        graph.num_neutral_PU = num_neutral_PU
        graph.num_neutral_LV = num_neutral_LV

    print("avg number of nodes " + str(avg_num_nodes / len(dataset)))
    print("avg number of edges " + str(avg_num_edges / len(dataset)))
    print("avg number of neutral LV " + str(avg_num_neu_LV / len(dataset)))
    print("avg number of neutral PU " + str(avg_num_neu_PU / len(dataset)))
    print("avg number of charged LV " + str(avg_num_chg_LV / len(dataset)))
    print("avg number of charged PU " + str(avg_num_chg_PU / len(dataset)))
    return dataset


def main():
    args = arg_parse()
    print(args.model_type)

    # load the constructed graphs
    with open(args.training_path, "rb") as fp:
        dataset = pickle.load(fp)
    with open(args.validation_path, "rb") as fp:
        dataset_validation = pickle.load(fp)

    train(dataset, dataset_validation, args, 1)


if __name__ == '__main__':
    main()
