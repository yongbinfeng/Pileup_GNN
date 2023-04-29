import math
from tqdm import tqdm
from collections import OrderedDict
from timeit import default_timer as timer
import pickle
import random
import numpy as np
import argparse
import torch
from torch_geometric.data import DataLoader
import models as models
import utils
import test_physics_metrics as phym
import matplotlib
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import mplhep as hep

hep.set_style(hep.style.CMS)

matplotlib.use("pdf")

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
    parser.add_argument('--training_path', type=str,
                        help='path for training graphs')
    parser.add_argument('--validation_path', type=str,
                        help='path for validation graphs')
    parser.add_argument('--save_dir', type=str,
                        help='directory to save trained model and plots')

    parser.set_defaults(model_type='Gated',
                        num_layers=3,
                        batch_size=4,
                        hidden_dim=20,
                        dropout=0,
                        opt='adam',
                        weight_decay=0,
                        lr=0.001,
                        pulevel=80,
                        training_path="../data_pickle/dataset_graph_puppi_20000",
                        validation_path="../data_pickle/dataset_graph_puppi_val_4000",
                        save_dir="test",
                        )

    return parser.parse_args()


def train(dataset, dataset_validation, args, batchsize):
    directory = args.save_dir
    # parent_dir = "/home/gpaspala/new_Pileup_GNN/Pileup_GNN/fast_simulation/"
    # path = os.path.join(parent_dir, directory)
    path = directory
    isdir = os.path.isdir(path)

    if isdir == False:
        os.mkdir(path)

    start = timer()

    rotate_mask = 5
    # rotate_mask = 2
    if args.pulevel == 20:
        rotate_mask = 8
        num_select_LV = 3
        num_select_PU = 45
    elif args.pulevel == 80:
        num_select_LV = 5
        num_select_PU = 11
    else:
        num_select_LV = 6
        num_select_PU = 282

    generate_mask(dataset, rotate_mask, num_select_LV, num_select_PU)
    generate_mask(dataset_validation, 1, num_select_LV, num_select_PU)

    training_loader = DataLoader(dataset, batch_size=batchsize)
    validation_loader = DataLoader(dataset_validation, batch_size=batchsize)

    model = models.GNNStack(
        dataset[0].num_feature_actual, args.hidden_dim, 1, args)
    model.to(device)
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    

    # train
    #
    # todo: this bunch of lists can be replaced with a small class or so
    #
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

    train_graph_SSLMassdiffMu = []
    train_graph_PUPPIMassdiffMu = []
    train_graph_SSLMassSigma = []
    train_graph_PUPPIMassSigma = []
    train_graph_SSLPtdiffMu = []
    train_graph_SSLPtSigma = []
    train_graph_PUPPIPtdiffMu = []
    train_graph_PUPPIPtSigma = []

    valid_graph_SSLMassdiffMu = []
    valid_graph_PUPPIMassdiffMu = []
    valid_graph_SSLMassSigma = []
    valid_graph_PUPPIMassSigma = []
    valid_graph_SSLPtdiffMu = []
    valid_graph_SSLPtSigma = []
    valid_graph_PUPPIPtdiffMu = []
    valid_graph_PUPPIPtSigma = []

    count_event = 0
    best_validation_auc = 0
    converge = False
    converge_num_event = 0
    last_steady_event = 0
    lowest_valid_loss = 10

    while converge == False:
        model.train()
        train_mask_all = None

        t = tqdm(total=len(training_loader), colour='green',
                 bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        loss_avg = utils.RunningAverage()
        for batch in training_loader:
            count_event += 1
            epochs_train.append(count_event)
            cur_loss = 0.
            feature_with_mask = batch.x
            for iter in range(rotate_mask):
                num_feature = batch.num_feature_actual[0].item()
                # print("num_feature ", num_feature)
                batch.x = torch.cat((feature_with_mask[:, 0:num_feature],
                                     feature_with_mask[:,
                                                       (num_feature + iter)].view(-1, 1),
                                     feature_with_mask[:, -num_feature:]), 1)
                batch = batch.to(device)

                pred, _ = model.forward(batch)

                label = batch.y
                train_mask = batch.x[:, num_feature]
                # print("train mask: ", torch.sum(train_mask))
                if train_mask_all != None:
                    train_mask_all = torch.cat((train_mask_all, train_mask), 0)
                else:
                    train_mask_all = train_mask

                label = label[train_mask == 1]
                label = label.type(torch.float)
                label = label.view(-1, 1)
                pred = pred[train_mask == 1]

                # print("pred: ", pred)
                # print("label: ", label)
                loss = model.loss(pred, label)
                cur_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()

            if math.isnan(cur_loss):
                print("cur_loss ", cur_loss)
                print("label: ", label)
                print("pred: ", pred)
            cur_loss = cur_loss / rotate_mask
            loss_graph.append(cur_loss)
            # print("cur_loss ", cur_loss)
            loss_avg.update(cur_loss)
            # print("loss_avg ", loss_avg())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

            if count_event % 1000 == 0:

                modelcolls = OrderedDict()
                modelcolls['gated_boost'] = model
                training_loss, training_loss_hybrid, train_acc, train_auc, train_auc_hybrid, \
                    train_puppi_acc, train_puppi_auc, \
                    train_acc_neu, train_auc_neu, train_auc_neu_hybrid, \
                    train_puppi_acc_neu, train_puppi_auc_neu, train_fig_name, train_SSLMassdiffMu, \
                    train_SSLMassSigma, train_PUPPIMassdiffMu, train_PUPPIMassSigma, train_SSLPtdiffMu, train_SSLPtSigma, \
                    train_PUPPIPtdiffMu, train_PUPPIPtSigma = test(
                        training_loader, model, 0, count_event, args, modelcolls, args.training_path)

                valid_loss, valid_loss_hybrid, valid_acc, valid_auc, valid_auc_hybrid, \
                    valid_puppi_acc, valid_puppi_auc, \
                    valid_acc_neu, valid_auc_neu, valid_auc_neu_hybrid, \
                    valid_puppi_acc_neu, valid_puppi_auc_neu, valid_fig_name, valid_SSLMassdiffMu, \
                    valid_SSLMassSigma, valid_PUPPIMassdiffMu, valid_PUPPIMassSigma, valid_SSLPtdiffMu, valid_SSLPtSigma, \
                    valid_PUPPIPtdiffMu, valid_PUPPIPtSigma  = test(
                        validation_loader, model, 1, count_event, args, modelcolls, args.validation_path)

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

                train_graph_SSLMassdiffMu.append(train_SSLMassdiffMu)
                train_graph_PUPPIMassdiffMu.append(train_PUPPIPtdiffMu)
                train_graph_SSLMassSigma.append(train_SSLMassSigma)
                train_graph_PUPPIMassSigma.append(train_PUPPIMassSigma)
                train_graph_SSLPtdiffMu.append(train_SSLPtdiffMu)
                train_graph_SSLPtSigma.append(train_SSLPtSigma)
                train_graph_PUPPIPtdiffMu.append(train_PUPPIPtdiffMu)
                train_graph_PUPPIPtSigma.append(train_PUPPIPtSigma)

                valid_graph_SSLMassdiffMu.append(valid_SSLMassdiffMu)
                valid_graph_PUPPIMassdiffMu.append(valid_PUPPIMassdiffMu)
                valid_graph_SSLMassSigma.append(valid_SSLMassSigma)
                valid_graph_PUPPIMassSigma.append(valid_PUPPIMassSigma)
                valid_graph_SSLPtdiffMu.append(valid_SSLPtdiffMu)
                valid_graph_SSLPtSigma.append(valid_SSLPtSigma)
                valid_graph_PUPPIPtdiffMu.append(valid_PUPPIPtdiffMu)
                valid_graph_PUPPIPtSigma.append(valid_PUPPIPtSigma)

                if valid_auc > best_validation_auc:
                    best_validation_auc = valid_auc
                    print("model is saved in " + path + "/best_valid_model_nPU11_deeper.pt")
                    torch.save(model.state_dict(), path +
                               "/best_valid_model_nPU11_deeper.pt")

                if valid_loss >= lowest_valid_loss:
                    print(
                        "valid loss increase at event " + str(count_event) + " with validation loss " + str(valid_loss))
                    if last_steady_event == count_event - 1000:
                        converge_num_event += 1
                        if converge_num_event > 30:
                            converge = True
                            break
                        else:
                            last_steady_event = count_event
                    else:
                        converge_num_event = 1
                        last_steady_event = count_event
                    # print("converge num event " + str(converge_num_event))
                else:
                    print("lowest valid loss " + str(valid_loss))
                    lowest_valid_loss = valid_loss

                if count_event == 5000:
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
                        auc_graph_neu_valid, auc_graph_valid_puppi_neu, dir_name=args.save_dir
                        )
    plt.figure()
    plt.plot(epochs_valid, train_graph_SSLMassdiffMu, label = 'Semi-supervised_train_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_valid, valid_graph_SSLMassdiffMu, label = 'Semi-supervised_valid_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'b')
    plt.plot(epochs_valid, train_graph_PUPPIMassdiffMu, label = 'PUPPI_train_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'r')
    #plt.plot(epochs_valid, valid_graph_PUPPIMassdiffMu, label = 'PUPPI_valid_JetMass, $\mu$', linestyle = 'solid', linewidth = 1, color = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('mean diff')
    plt.legend(loc=4)
    plt.savefig(args.save_dir + "/Jet_mass_diff_mean.pdf")
    plt.close()

    plt.figure()
    plt.plot(epochs_valid, train_graph_SSLPtdiffMu, label = 'Semi-supervised_train_JetPt, $\mu$', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_valid, valid_graph_SSLPtdiffMu, label = 'Semi-supervised_valid_JetPt, $\mu$', linestyle = 'solid', linewidth = 1, color = 'b')
    plt.plot(epochs_valid, train_graph_PUPPIPtdiffMu, label = 'PUPPI_train_JetPt, $\mu$', linestyle = 'solid', linewidth = 1, color = 'r')
    #plt.plot(epochs_valid, valid_graph_PUPPIPtdiffMu, label = 'PUPPI_valid_JetPt, $\mu$', linestyle = 'solid', linewidth = 1, color = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('mean diff')
    plt.legend(loc=4)
    plt.savefig(args.save_dir + "/Jet_pt_diff_mean.pdf")
    plt.close()

    plt.figure()
    plt.plot(epochs_valid, train_graph_SSLMassSigma, label = 'Semi-supervised_train_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_valid, valid_graph_SSLMassSigma, label = 'Semi-supervised_valid_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'b')
    plt.plot(epochs_valid, train_graph_PUPPIMassSigma, label = 'PUPPI_train_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'r')
    #plt.plot(epochs_valid, valid_graph_PUPPIMassSigma, label = 'PUPPI_valid_JetMass, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('sigma diff')
    plt.legend(loc=4)
    plt.savefig(args.save_dir + "/Jet_mass_diff_sigma.pdf")
    plt.close()

    plt.figure()
    plt.plot(epochs_valid, train_graph_SSLPtSigma, label = 'Semi-supervised_train_JetPt, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_valid, valid_graph_SSLPtSigma, label = 'Semi-supervised_valid_JetPt, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'b')
    plt.plot(epochs_valid, train_graph_PUPPIPtSigma, label = 'PUPPI_train_JetPt, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'r')
    #plt.plot(epochs_valid, valid_graph_PUPPIPtSigma, label = 'PUPPI_valid_JetPt, $\sigma$', linestyle = 'solid', linewidth = 1, color = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('sigma diff')
    plt.legend(loc=4)
    plt.savefig(args.save_dir + "/Jet_pt_diff_sigma.pdf")
    plt.close()

    # utils.plot_training(epochs_train, epochs_valid, loss_graph_train,
    #                   loss_graph, auc_graph_train, train_accuracy_neu,
    #                   loss_graph_valid, auc_graph_valid, valid_accuracy_neu,
    #                   auc_graph_neu_train,
    #                   auc_graph_neu_valid, dir_name = args.save_dir
    #                   )


def test(loader, model, indicator, epoch, args, modelcolls, pathname):
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

            data.x = torch.cat(
                (data.x[:, 0:num_feature], test_mask.view(-1, 1), data.x[:, -num_feature:]), 1)
            data = data.to(device)
            # max(dim=1) returns values, indices tuple; only need indices
            pred, pred_hybrid = model.forward(data)
            # puppi = data.x[:, data.num_feature_actual[0].item() - 1]
            puppi = data.pWeight
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
            total_loss_hybrid += model.loss(pred_hybrid,
                                            label).item() * data.num_graphs

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

    utils.plot_roc([label_all_chg],
                   [pred_all_chg],
                   legends=["prediction Chg"],
                   postfix=postfix + "_test", dir_name=args.save_dir)

    fig_name_prediction = utils.plot_discriminator(epoch,
                                                   [pred_all_chg[label_all_chg == 1], pred_all_chg[label_all_chg == 0],
                                                    pred_all_neu[label_all_neu == 1],
                                                    pred_all_neu[label_all_neu == 0]],
                                                   legends=[
                                                       'LV Chg', 'PU Chg', 'LV Neu', 'PU Neu'],
                                                   postfix=postfix + "_prediction", label='Prediction', dir_name=args.save_dir)
    fig_name_puppi = utils.plot_discriminator(epoch,
                                              [puppi_all_chg[label_all_chg == 1], puppi_all_chg[label_all_chg == 0],
                                               puppi_all_neu[label_all_neu == 1],
                                               puppi_all_neu[label_all_neu == 0]],
                                              legends=[
                                                  'LV Chg', 'PU Chg', 'LV Neu', 'PU Neu'],
                                              postfix=postfix + "_puppi", label='PUPPI Weight', dir_name=args.save_dir)
    
    filelists = []
    filelists.append(pathname)

    mets_truth, performances_jet_CHS, performances_jet_puppi, mets_puppi, performances_jet_puppi_wcut, mets_puppi_wcut, performances_jet_pred, mets_pred, neu_weight, neu_puppiweight, chlv_weight, chpu_weight, chlv_puppiweight, chpu_puppiweight, njets_pf, njets_pred, njets_puppi, njets_truth, njets_CHS, pt_jets_pf, pt_jets_pred, pt_jets_puppi, pt_jets_truth, pt_jets_CHS, eta_jets_pf, eta_jets_pred, eta_jets_puppi, eta_jets_truth, eta_jets_CHS, phi_jets_pf, phi_jets_pred, phi_jets_puppi, phi_jets_truth, phi_jets_CHS, mass_jets_pf, mass_jets_pred, mass_jets_puppi, mass_jets_truth, mass_jets_CHS =phym.test(
        filelists, modelcolls)

    # plot the differences
    def getResol(input):
        return (np.quantile(input, 0.84) - np.quantile(input, 0.16))/2

    def getStat(input):
        return float(np.median(input)), float(getResol(input))

    performances_jet_pred0 = performances_jet_pred['gated_boost']
    # performances_jet_pred4 = performances_jet_pred['gated_boost_sp']

    mets_pred0 = mets_pred['gated_boost']
    # mets_pred4 = mets_pred['gated_boost_sp']

    linewidth = 1.5
    fontsize = 18

    plt.style.use(hep.style.ROOT)
    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_pred0])
    print(mass_diff)
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='blue', linewidth=linewidth,
             density=True, label=r'Semi-supervised, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    SSLMassdiffMu, SSLMassSigma = getStat(mass_diff)
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_puppi])
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='green', linewidth=linewidth, 
             density=True, label=r'PUPPI, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    PUPPIMassdiffMu, PUPPIMassSigma = getStat(mass_diff)
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_puppi_wcut])
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='red', linewidth=linewidth, 
             density=True, label=r'PF, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_CHS])
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='orange', linewidth=linewidth, 
             density=True, label=r'CHS, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    # plt.xlim(-1.0,1.3)
    plt.xlabel(r"Jet Mass $(m_{reco} - m_{truth})/m_{truth}$ Epoch"+str(epoch)+postfix)
    plt.ylabel('density')
    plt.ylim(0, 3.6)
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.savefig(args.save_dir+"/prob_plots/Jet_mass_diff_"+postfix+str(epoch)+".pdf")
    plt.show()

    fig = plt.figure(figsize=(10, 8))

    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_pred0])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='blue', linewidth=linewidth, 
             density=True, label=r'Semi-supevised, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pt_diff)))+str(len(pt_diff)))
    SSLPtdiffMu, SSLPtSigma = getStat(pt_diff)
    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_puppi])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='green', linewidth=linewidth, 
             density=True, label=r'PUPPI, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pt_diff)))+str(len(pt_diff)))
    PUPPIPtdiffMu, PUPPIPtSigma = getStat(pt_diff)
    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_puppi_wcut])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='red', linewidth=linewidth, 
             density=True, label=r'PF, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pt_diff)))+str(len(pt_diff)))
    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_CHS])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='orange', linewidth=linewidth, 
             density=True, label=r'CHS, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pt_diff)))+str(len(pt_diff)))
    # plt.xlim(0,40)
    plt.ylim(0, 7)
    plt.xlabel(r"Jet $p_{T}$ $(p^{reco}_{T} - p^{truth}_{T})/p^{truth}_{T}$ Epoch"+str(epoch)+postfix)
    plt.ylabel('density')
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.show()
    plt.savefig(args.save_dir+"/prob_plots/Jet_pT_diff_"+postfix+str(epoch)+".pdf")

    return total_loss, total_loss_hybrid, acc_chg, auc_chg, auc_chg_hybrid, acc_chg_puppi, auc_chg_puppi, \
        acc_neu, auc_neu, auc_neu_hybrid, acc_neu_puppi, auc_neu_puppi, fig_name_prediction, SSLMassdiffMu, \
        SSLMassSigma, PUPPIMassdiffMu, PUPPIMassSigma, SSLPtdiffMu, SSLPtSigma, PUPPIPtdiffMu, PUPPIPtSigma


def generate_mask(dataset, num_mask, num_select_LV, num_select_PU):
    # how many LV and PU to sample
    for graph in dataset:
        LV_index = graph.LV_index
        PU_index = graph.PU_index
        np.random.shuffle(LV_index)
        np.random.shuffle(PU_index)
        original_feature = graph.x[:, 0:graph.num_feature_actual]

        # pf_dz_training = torch.zeros(graph.num_nodes, num_mask)
        mask_training = torch.zeros(graph.num_nodes, num_mask)
        for num in range(num_mask):
            if LV_index.shape[0] < num_select_LV or PU_index.shape[0] < num_select_PU:
                num_select_LV = min(LV_index.shape[0], num_select_LV)
                num_select_PU = min(PU_index.shape[0], num_select_PU)

            # generate the index for LV and PU samples for training mask
            # gen_index_LV = random.sample(range(LV_index.shape[0]), num_select_LV)
            selected_LV_train = np.take(LV_index, range(
                num * num_select_LV, (num + 1) * num_select_LV), mode='wrap')

            # gen_index_PU = random.sample(range(PU_index.shape[0]), num_select_PU)
            selected_PU_train = np.take(PU_index, range(
                num * num_select_PU, (num + 1) * num_select_PU), mode='wrap')

            training_mask = np.concatenate(
                (selected_LV_train, selected_PU_train), axis=None)
            # print(training_mask)

            # construct mask vector for training and testing
            mask_training_cur = torch.zeros(graph.num_nodes)
            mask_training_cur[[training_mask.tolist()]] = 1
            mask_training[:, num] = mask_training_cur

        x_concat = torch.cat((original_feature, mask_training), 1)
        graph.x = x_concat

        # mask the puppiWeight as default Neutral(here puppiweight is actually fromLV in ggnn dataset)
        puppiWeight_default_one_hot_training = torch.cat((torch.zeros(graph.num_nodes, 1),
                                                          torch.zeros(
                                                              graph.num_nodes, 1),
                                                          torch.ones(graph.num_nodes, 1)), 1)
        puppiWeight_default_one_hot_training = puppiWeight_default_one_hot_training.type(
            torch.float32)

        # mask the pdgID for charge particles
        pdgId_one_hot_training = torch.cat((torch.zeros(graph.num_nodes, 1),
                                            torch.zeros(graph.num_nodes, 1),
                                            torch.ones(graph.num_nodes, 1)), 1)
        pdgId_one_hot_training = pdgId_one_hot_training.type(torch.float32)

        # pf_dz_training_test = torch.clone(original_feature[:, 6:7])
        # print ("pf_dz_training_test: ", pf_dz_training_test)
        # print ("pf_dz_training_test: ", pf_dz_training_test.shape)
        # pf_dz_training_test[[training_mask.tolist()],0]=0
        # pf_dz_training_test = torch.zeros(graph.num_nodes, 1)

        # print ("pf_dz_training_test: ", pf_dz_training_test)
        # print ("puppiWeight_default_one_hot_training size: ", puppiWeight_default_one_hot_training.size())
        # print ("pf_dz_training_test size: ", pf_dz_training_test.size())

        # print ("pf_dz_training_test: ", pf_dz_training_test)

        # replace the one-hot encoded puppi weights and PF_dz
        # default_data_training = torch.cat(
        #   (original_feature[:, 0:(graph.num_feature_actual - 3)], puppiWeight_default_one_hot_training), 1)

        # default_data_training = torch.cat(
        #    (original_feature[:, 0:(graph.num_feature_actual - 4)],pf_dz_training_test ,puppiWeight_default_one_hot_training), 1)
        # add masked PdgID
        default_data_training = torch.cat(
            (original_feature[:, 0:(graph.num_feature_actual - 6)], pdgId_one_hot_training, puppiWeight_default_one_hot_training), 1)

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
        Neutral_index = Neutral_index[torch.where(
            Neutral_feature[:, 2] > 0.5)[0]]

        mask_neu = torch.zeros(nparticles, 1)
        mask_neu[Neutral_index, 0] = 1
        graph.mask_neu = mask_neu

    return dataset


def main():
    args = arg_parse()
    print("model type: ", args.model_type)

    # load the constructed graphs
    with open(args.training_path, "rb") as fp:
        dataset = pickle.load(fp)
    with open(args.validation_path, "rb") as fp:
        dataset_validation = pickle.load(fp)

    generate_neu_mask(dataset)
    generate_neu_mask(dataset_validation)
    train(dataset, dataset_validation, args, 1)


if __name__ == '__main__':
    main()
