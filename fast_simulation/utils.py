import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib as mpl
import imageio
import matplotlib.pyplot as plt
import mplhep as hep
mpl.use("pdf")
hep.set_style(hep.style.ROOT)
import numpy as np
import pickle
from copy import deepcopy
import random
import os

# num graphs, lr, num_layer, hidden_neuron, change_mask, epoch, model, loss
dir_name = "Gated_PU80_r08_007_2_20_sup_noboost/"


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:se
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
                            help='Optimizer weight decay.')


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    scheduler = None
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def generate_random_mask(nparticles, nselect, LV_index, PU_index):
    """
    randomly select nselect particles from LV_index and
    nselect particles from PU_index,
    return the indices of the selected
    """
    nselect = min(LV_index.shape[0], PU_index.shape[0], nselect)

    # generate the index for LV and PU samples for training mask
    gen_index_LV = random.sample(range(LV_index.shape[0]), nselect)
    selected_LV_train = LV_index[gen_index_LV]

    gen_index_PU = random.sample(range(PU_index.shape[0]), nselect)
    selected_PU_train = PU_index[gen_index_PU]

    training_mask = np.concatenate((selected_LV_train, selected_PU_train), axis=None)

    # construct mask vector for training and testing
    mask_training = torch.zeros(nparticles, 1)
    mask_training[training_mask, 0] = 1

    return mask_training


def get_acc(truth, prediction, cut=0.5):
    truth = truth.astype('int32')
    predict = np.copy(prediction)
    predict[predict > cut] = 1
    predict[predict < cut] = 0
    predict = predict.astype('int32')
    acc = accuracy_score(truth, predict)
    return acc


def get_auc(truth, prediction):
    auc = roc_auc_score(truth, prediction)
    return auc


def plot_roc(truths, predictions, legends, postfix="", saveTo=None):
    """
    plot the roc based on the truth (label), and the list of predictions
    """
    colors = ['b', 'g', 'r', 'm']
    plt.figure()
    for i in range(len(predictions)):
        truth = truths[i]
        prediction = predictions[i]
        legend = legends[i]

        fpr, tpr, thresholds = roc_curve(truth, prediction)
        auc = get_auc(truth, prediction)
        plt.plot(fpr, tpr, label=legend + ", auc=" + np.format_float_positional(auc * 100, precision=2) + "%",
                 linestyle='solid', linewidth=2, color=colors[i])
        # print(legend, 'auc (%)', auc*100)

        # Save to file if needed
        if saveTo is not None:
            file = open(saveTo, 'wb')
            pickle.dump([fpr, tpr], file)
            file.close()

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.grid()
    plt.legend(loc=4, fontsize=28)
    plt.savefig(dir_name + "roc_" + postfix + ".pdf", bbox_inches='tight')
    plt.close()

def plot_roc_logscale(truths, predictions, legends, postfix="", saveTo=None):
    """
    plot the roc based on the truth (label), and the list of predictions
    """
    colors = ['b', 'g', 'r', 'm']
    plt.figure()
    for i in range(len(predictions)):
        truth = truths[i]
        prediction = predictions[i]
        legend = legends[i]

        fpr, tpr, thresholds = roc_curve(truth, prediction)
        auc = get_auc(truth, prediction)
        plt.plot(fpr, tpr, label=legend + ", auc=" + np.format_float_positional(auc * 100, precision=2) + "%",
                 linestyle='solid', linewidth=2, color=colors[i])

        # Save to file if needed
        if saveTo is not None:
            file = open(saveTo, 'wb')
            pickle.dump([fpr, tpr], file)
            file.close()

    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    plt.xlim([1e-4, 0.01])
    plt.xscale('log')
    plt.ylim([1e-4, 1])
    plt.yscale('log')
    plt.grid()
    #plt.legend(loc=4, fontsize=25)
    plt.savefig(dir_name + "roc_logscale_cut" + postfix + ".pdf", bbox_inches='tight')
    plt.close()

def plot_roc_lowerleft(truths, predictions, legends, postfix="", saveTo=None):
    """
    plot the roc based on the truth (label), and the list of predictions
    """
    colors = ['b', 'g', 'r', 'm']
    plt.figure()
    max_tpr = 0
    for i in range(len(predictions)):
        truth = truths[i]
        prediction = predictions[i]
        legend = legends[i]

        fpr, tpr, thresholds = roc_curve(truth, prediction)
        truncate_index = np.where(fpr < 0.2)[0].size
        fpr_truncate = fpr[0:truncate_index]
        tpr_truncate = tpr[0:truncate_index]
        cur_tpr = np.max(tpr_truncate)
        if cur_tpr > max_tpr:
            max_tpr = cur_tpr
        auc = get_auc(truth, prediction)
        plt.plot(fpr_truncate, tpr_truncate, label=legend + ", auc=" + np.format_float_positional(auc * 100, precision=2) + "%",
                 linestyle='solid', linewidth=2, color=colors[i])
        # print(legend, 'auc (%)', auc*100)

        # Save to file if needed
        if saveTo is not None:
            file = open(saveTo, 'wb')
            pickle.dump([fpr, tpr], file)
            file.close()

    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    plt.grid()
    #plt.legend(loc=4)
    plt.savefig(dir_name + "lowerleft_roc_" + postfix + ".pdf", bbox_inches='tight')
    plt.close()

def plot_hist2d(pts, weights, yname='Weight', postfix=""):
    """
    plot the (puppi)Weights vs pt 2D histogram
    """
    plt.figure()
    h = plt.hist2d(pts, weights, bins=20, range=[[0, 4.0], [0, 1.0]], norm=mpl.colors.LogNorm())
    plt.colorbar(h[3])
    plt.xlabel('p_{T} [GeV]')
    plt.ylabel(yname)
    plt.grid()
    plt.savefig(dir_name + "hist2d_pt_vs_" + yname.replace(' ', '') + "_" + postfix + ".pdf")
    plt.close()


def plot_discriminator(epoch, vals, legends=['LV', 'PU'], postfix="", label="Discriminator", bins=50, xaxisrange=(0, 1)):
    """
    plot the distriminator distribution
    """
    sub_dir = "prob_plots"
    parent_dir = "/home/liu2112/project/" + dir_name
    path = os.path.join(parent_dir, sub_dir)
    isdir = os.path.isdir(path)
    if isdir == False:
        os.mkdir(path)

    plt.figure()
    for i in range(len(vals)):
        val = vals[i]
        legend = legends[i]
        plt.hist(val, bins=50, range=xaxisrange, density=True, histtype='step', label=legend)
    plt.ylabel('A.U.')
    plt.xlabel(label + str(epoch))
    plt.legend(loc=4)
    filename = dir_name + "prob_plots/Distriminator_" + postfix + "_" + str(epoch) +".png"
    plt.savefig(filename)
    plt.close()

    return filename


def plot_training(
        epochs_train, epochs_test, loss_graph_train,
        loss_graph, auc_graph_train, train_accuracy, auc_graph_train_puppi, train_accuracy_puppi,
        loss_graph_test, auc_graph_test, test_accuracy, auc_graph_test_puppi, test_accuracy_puppi,
        auc_graph_neu_train, auc_graph_train_puppi_neu,
        auc_graph_neu_test, auc_graph_test_puppi_neu,
        postfix=".pdf"
):
    # print(epochs_train)
    # print(epochs_test)
    # print(loss_graph)
    # print(auc_graph_test)

    # loss
    plt.figure()
    #plt.plot(epochs_train, loss_graph, label='train', linestyle='solid', linewidth=1, color='r')
    plt.plot(epochs_test, loss_graph_train, label = 'train_avg', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_test, loss_graph_test, label='Validation', linestyle='solid', linewidth=1, color='b')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend(loc=4)
    plt.savefig(dir_name + "loss_graph" + postfix)
    plt.close()

    # auc
    plt.figure()
    plt.plot(epochs_test, auc_graph_train, label="train", linestyle='solid', linewidth=1, color='r')
    plt.plot(epochs_test, auc_graph_test, label="Validation", linestyle='solid', linewidth=1, color='b')
    plt.plot(epochs_test, auc_graph_train_puppi, label="PUPPI train", linestyle='dashed', linewidth=1,
             color='g')
    plt.plot(epochs_test, auc_graph_test_puppi, label="PUPPI Validation", linestyle='solid', linewidth=1,
             color='g')
    plt.plot(epochs_test, auc_graph_neu_train, label="Neu train", linestyle='solid', linewidth=1,
             color='orange')
    plt.plot(epochs_test, auc_graph_neu_test, label="neu Validation", linestyle='solid', linewidth=1, color='cyan')
    plt.plot(epochs_test, auc_graph_train_puppi_neu, label="PUPPI Neu train", linestyle='dashed',
             linewidth=1, color='m')
    plt.plot(epochs_test, auc_graph_test_puppi_neu, label="PUPPI Neu Validation", linestyle='solid', linewidth=1,
             color='m')
    plt.xlabel('Epochs')
    plt.ylabel('auc')
    plt.legend(loc=4)
    plt.savefig(dir_name + "auc_graph_train" + postfix)
    plt.close()

    # accuracy
    plt.figure()
    plt.plot(epochs_test, train_accuracy, label="train", linestyle='solid', linewidth=1, color='r')
    plt.plot(epochs_test, test_accuracy, label="Validation", linestyle='solid', linewidth=1, color='b')
    plt.plot(epochs_test, train_accuracy_puppi, label="PUPPI train", linestyle='dashed', linewidth=1,
             color='g')
    plt.plot(epochs_test, test_accuracy_puppi, label="PUPPI Validation", linestyle='solid', linewidth=1,
             color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc=4)
    plt.savefig(dir_name + "accuracy_graph" + postfix)
    plt.clf()
    plt.close()

def plot_training_fullsim(
        epochs_train, epochs_test, loss_graph_train,
        loss_graph, auc_graph_train, train_accuracy, auc_graph_train_puppi, train_accuracy_puppi,
        loss_graph_test, auc_graph_test, test_accuracy, auc_graph_test_puppi, test_accuracy_puppi,
        postfix=".pdf"
):
    # print(epochs_train)
    # print(epochs_test)
    # print(loss_graph)
    # print(auc_graph_test)

    # loss
    plt.figure()
    #plt.plot(epochs_train, loss_graph, label='train', linestyle='solid', linewidth=1, color='r')
    plt.plot(epochs_test, loss_graph_train, label = 'train_avg', linestyle = 'solid', linewidth = 1, color = 'g')
    plt.plot(epochs_test, loss_graph_test, label='Validation', linestyle='solid', linewidth=1, color='b')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend(loc=4)
    plt.savefig(dir_name + "loss_graph" + postfix)
    plt.close()

    # auc
    plt.figure()
    plt.plot(epochs_test, auc_graph_train, label="train", linestyle='solid', linewidth=1, color='r')
    plt.plot(epochs_test, auc_graph_test, label="Validation", linestyle='solid', linewidth=1, color='b')
    plt.plot(epochs_test, auc_graph_train_puppi, label="PUPPI train", linestyle='dashed', linewidth=1,
             color='g')
    plt.plot(epochs_test, auc_graph_test_puppi, label="PUPPI Validation", linestyle='solid', linewidth=1,
             color='g')
    plt.xlabel('Epochs')
    plt.ylabel('auc')
    plt.legend(loc=4)
    plt.savefig(dir_name + "auc_graph_train" + postfix)
    plt.close()

    # accuracy
    plt.figure()
    plt.plot(epochs_test, train_accuracy, label="train", linestyle='solid', linewidth=1, color='r')
    plt.plot(epochs_test, test_accuracy, label="Validation", linestyle='solid', linewidth=1, color='b')
    plt.plot(epochs_test, train_accuracy_puppi, label="PUPPI train", linestyle='dashed', linewidth=1,
             color='g')
    plt.plot(epochs_test, test_accuracy_puppi, label="PUPPI Validation", linestyle='solid', linewidth=1,
             color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc=4)
    plt.savefig(dir_name + "accuracy_graph" + postfix)
    plt.clf()
    plt.close()

def plot_testing(epochs_test,
        loss_graph_test, auc_graph_test, test_accuracy, auc_graph_test_puppi, test_accuracy_puppi,
        ):
    # loss
    postfix = "_test.pdf"
    plt.figure()
    plt.plot(epochs_test, loss_graph_test, label='test', linestyle='solid', linewidth=1, color='b')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend(loc=4)
    plt.savefig(dir_name + "loss_graph" + postfix)
    plt.close()

    # auc
    plt.figure()
    plt.plot(epochs_test, auc_graph_test, label="test", linestyle='solid', linewidth=1, color='b')
    plt.plot(epochs_test, auc_graph_test_puppi, label="PUPPI test", linestyle='solid', linewidth=1,
             color='g')
    plt.xlabel('Epochs')
    plt.ylabel('auc')
    plt.legend(loc=4)
    plt.savefig(dir_name + "auc_graph_train" + postfix)
    plt.close()

    # accuracy
    plt.figure()
    plt.plot(epochs_test, test_accuracy, label="test", linestyle='solid', linewidth=1, color='b')
    plt.plot(epochs_test, test_accuracy_puppi, label="PUPPI test", linestyle='solid', linewidth=1,
             color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc=4)
    plt.savefig(dir_name + "accuracy_graph" + postfix)
    plt.clf()
    plt.close()

def plot_kinematics(dataset):
    """
    plot the kinematic distribution of given particles
    """
    pts = None
    etas = None
    phis = None
    chgs = None
    fromLVs = None
    weights = None
    chgMasks = None
    neuMasks = None

    isfirst = True
    for graph in dataset:
        num_features = graph.num_feature_actual
        features = graph.x.cpu().numpy()
        mask = features[:, num_features]
        mask_neu = graph.mask_neu[:, 0].cpu().numpy()
        pt = features[:, 2]
        eta = features[:, 0]
        phi = features[:, 1]
        chg = features[:, 3]
        fromLV = features[:, 4]
        weight = features[:, 5]

        if not isfirst:
            chgMasks = np.concatenate((chgMasks, mask), 0)
            neuMasks = np.concatenate((neuMasks, mask_neu), 0)
            pts = np.concatenate((pts, pt), 0)
            etas = np.concatenate((etas, eta), 0)
            phis = np.concatenate((phis, phi), 0)
            chgs = np.concatenate((chgs, chg), 0)
            fromLVs = np.concatenate((fromLVs, fromLV), 0)
            weights = np.concatenate((weights, weight), 0)
        else:
            chgMasks = mask
            neuMasks = mask_neu
            pts = pt
            etas = eta
            phis = phi
            chgs = chg
            fromLVs = fromLV
            weights = weight
            isfirst = False

    chgMasks = chgMasks.astype(int)
    neuMasks = neuMasks.astype(int)

    plt.figure()
    plt.hist(pts[chgMasks == 1], bins=50, density=True, histtype='step', label='Chg')
    plt.hist(pts[neuMasks == 1], bins=50, density=True, histtype='step', label='Neu')
    plt.ylabel('A.U.')
    plt.yscale('log')
    plt.xlabel('p_{T} [GeV]')
    plt.legend(loc=4)
    plt.savefig(dir_name + "pt.pdf")
    plt.close()

    plt.figure()
    plt.hist(etas[chgMasks == 1], bins=50, density=True, histtype='step', label='Chg')
    plt.hist(etas[neuMasks == 1], bins=50, density=True, histtype='step', label='Neu')
    plt.ylabel('A.U.')
    plt.xlabel('eta')
    plt.legend(loc=4)
    plt.savefig(dir_name + "eta.pdf")
    plt.close()

    plt.figure()
    plt.hist(phis[chgMasks == 1], bins=50, density=True, histtype='step', label='Chg')
    plt.hist(phis[neuMasks == 1], bins=50, density=True, histtype='step', label='Neu')
    plt.ylabel('A.U.')
    plt.xlabel('phi')
    plt.legend(loc=4)
    plt.savefig(dir_name + "phi.pdf")
    plt.close()

    plt.figure()
    plt.hist(fromLVs[chgMasks == 1], bins=50, density=True, histtype='step', label='Chg')
    plt.hist(fromLVs[neuMasks == 1], bins=50, density=True, histtype='step', label='Neu')
    plt.ylabel('A.U.')
    plt.xlabel('fromLVs')
    plt.legend(loc=4)
    plt.savefig(dir_name + "fromLVs.pdf")
    plt.close()

    plt.figure()
    plt.hist(chgs[chgMasks == 1], bins=50, density=True, histtype='step', label='Chg')
    plt.hist(chgs[neuMasks == 1], bins=50, density=True, histtype='step', label='Neu')
    plt.ylabel('A.U.')
    plt.xlabel('charge')
    plt.legend(loc=4)
    plt.savefig(dir_name + "charge.pdf")
    plt.close()

    plt.figure()
    plt.hist(weights[chgMasks == 1], bins=50, density=True, histtype='step', label='Chg')
    plt.hist(weights[neuMasks == 1], bins=50, density=True, histtype='step', label='Neu')
    plt.ylabel('A.U.')
    plt.xlabel('weights')
    plt.legend(loc=4)
    plt.savefig(dir_name + "weights.pdf")
    plt.close()


def make_gif(figures, postfix="Train"):
    """
    make the fig based on the list of figures
    """
    with imageio.get_writer(dir_name + 'result_' + postfix + ".gif") as writer:
        for fig in figures:
            image = imageio.imread(fig)
            writer.append_data(image)

def plot_boost_visualization(label, pred, puppi, x, model, auc_neu_puppi, event_nodes, postfix, hybrid_bol):
    target_event = auc_neu_puppi.index(min(auc_neu_puppi))
    starting_chg_index = sum(event_nodes[0][0:target_event])
    ending_chg_index = starting_chg_index + event_nodes[0][target_event]
    starting_neu_index = sum(event_nodes[1][0:target_event])
    ending_neu_index = starting_neu_index + event_nodes[1][target_event]

    beta_one = model.state_dict()['beta_one']
    beta_one = torch.sigmoid(beta_one).item()
    beta_two = 1-beta_one

    pred_chg = pred[0][starting_chg_index:ending_chg_index]
    pred_neu = pred[1][starting_neu_index:ending_neu_index]
    puppi_chg = puppi[0][starting_chg_index:ending_chg_index]
    puppi_neu = puppi[1][starting_neu_index:ending_neu_index]
    label_chg = label[0][starting_chg_index:ending_chg_index]
    label_neu = label[1][starting_neu_index:ending_neu_index]

    if hybrid_bol == True:
        original_pred_chg = (pred_chg - beta_two * puppi_chg.reshape(-1, 1)) / beta_one
        original_pred_neu = (pred_neu - beta_two * puppi_neu.reshape(-1, 1)) / beta_one
    else:
        original_pred_chg = pred_chg
        original_pred_neu = pred_neu


    #boost_diff_neu = pred_neu - original_pred_neu

    """
    significant_indices = np.where(abs(boost_diff) > 0.01)[0]
    significant_diff = boost_diff[significant_indices]
    significant_label = label[significant_indices]
    significant_label[significant_label == 0] = -1
    temp = significant_label * significant_diff.reshape(-1)
    effective_indices = significant_indices[np.where(temp > 0)[0]]
    """

    Neutral_LV = np.where(label_neu == 1)[0]
    Neutral_PU = np.where(label_neu == 0)[0]
    Charge_LV = np.where(label_chg == 1)[0]
    Charge_PU = np.where(label_chg == 0)[0]

    eta_Neutral_LV = x[1][Neutral_LV, 0]
    eta_Neutral_PU = x[1][Neutral_PU, 0]
    phi_Neutral_LV = x[1][Neutral_LV, 1]
    phi_Neutral_PU = x[1][Neutral_PU, 1]
    pt_Neutral_LV = x[1][Neutral_LV, 2]
    pt_Neutral_PU = x[1][Neutral_PU, 2]

    eta_Charge_LV = x[0][Charge_LV, 0]
    eta_Charge_PU = x[0][Charge_PU, 0]
    phi_Charge_LV = x[0][Charge_LV, 1]
    phi_Charge_PU = x[0][Charge_PU, 1]
    pt_Charge_LV = x[0][Charge_LV, 2]
    pt_Charge_PU = x[0][Charge_PU, 2]

    pt_Neutral_LV_withpred = pt_Neutral_LV * original_pred_neu[Neutral_LV].reshape(-1)
    pt_Neutral_PU_withpred = pt_Neutral_PU * original_pred_neu[Neutral_PU].reshape(-1)
    pt_Neutral_LV_withboost = pt_Neutral_LV * pred_neu[Neutral_LV].reshape(-1)
    pt_Neutral_PU_withboost = pt_Neutral_PU * pred_neu[Neutral_PU].reshape(-1)
    pt_Neutral_LV_puppi = pt_Neutral_LV * puppi_neu[Neutral_LV].reshape(-1)
    pt_Neutral_PU_puppi = pt_Neutral_PU * puppi_neu[Neutral_PU].reshape(-1)

    pt_Neutral_LV_withpred[pt_Neutral_LV_withpred < 0.1] = 0
    pt_Neutral_PU_withpred[pt_Neutral_PU_withpred < 0.1] = 0
    pt_Neutral_LV_withboost[pt_Neutral_LV_withboost < 0.1] = 0
    pt_Neutral_PU_withboost[pt_Neutral_PU_withboost < 0.1] = 0
    pt_Neutral_LV_puppi[pt_Neutral_LV_puppi < 0.1] = 0
    pt_Neutral_PU_puppi[pt_Neutral_PU_puppi < 0.1] = 0

    pt_Charge_LV_withpred = pt_Charge_LV * original_pred_chg[Charge_LV].reshape(-1)
    pt_Charge_PU_withpred = pt_Charge_PU * original_pred_chg[Charge_PU].reshape(-1)
    pt_Charge_LV_withboost = pt_Charge_LV * pred_chg[Charge_LV].reshape(-1)
    pt_Charge_PU_withboost = pt_Charge_PU * pred_chg[Charge_PU].reshape(-1)
    pt_Charge_LV_puppi = pt_Charge_LV * puppi_chg[Charge_LV].reshape(-1)
    pt_Charge_PU_puppi = pt_Charge_PU * puppi_chg[Charge_PU].reshape(-1)

    pt_Charge_LV_withpred[pt_Charge_LV_withpred < 0.1] = 0
    pt_Charge_PU_withpred[pt_Charge_PU_withpred < 0.1] = 0
    pt_Charge_LV_withboost[pt_Charge_LV_withboost < 0.1] = 0
    pt_Charge_PU_withboost[pt_Charge_PU_withboost < 0.1] = 0
    pt_Charge_LV_puppi[pt_Charge_LV_puppi < 0.1] = 0
    pt_Charge_PU_puppi[pt_Charge_PU_puppi < 0.1] = 0


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(eta_Neutral_PU, phi_Neutral_PU, s=pt_Neutral_PU * 20.0, c='orange',
               label='PU Neu')
    ax.scatter(eta_Neutral_LV, phi_Neutral_LV, s=pt_Neutral_LV * 20.0, c='blue',
               label='LV Neu')
    ax.scatter(eta_Charge_PU, phi_Charge_PU, s=pt_Charge_PU * 20.0, c='red',
               label='PU Chg')
    ax.scatter(eta_Charge_LV, phi_Charge_LV, s=pt_Charge_LV * 20.0, c='green',
               label='LV Chg')

    ax.set_ylabel(r'$\phi$')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r'$\eta$')
    ax.set_xlim(-4, 4)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0.)
    plt.savefig(dir_name + "original_pt_" + postfix + ".pdf")
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(eta_Neutral_PU, phi_Neutral_PU, s=pt_Neutral_PU_withpred * 20.0, c='orange',
               label='PU Neu')
    ax.scatter(eta_Neutral_LV, phi_Neutral_LV, s=pt_Neutral_LV_withpred * 20.0, c='blue',
               label='LV Neu')
    ax.scatter(eta_Charge_PU, phi_Charge_PU, s=pt_Charge_PU_withpred * 20.0, c='red',
               label='PU Chg')
    ax.scatter(eta_Charge_LV, phi_Charge_LV, s=pt_Charge_LV_withpred * 20.0, c='green',
               label='LV Chg')
    ax.set_ylabel(r'$\phi$')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r'$\eta$')
    ax.set_xlim(-4, 4)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0.)
    plt.savefig(dir_name + "mitigated_pt_prediction_" + postfix +".pdf")
    plt.close()

    # plot puppi
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(eta_Neutral_PU, phi_Neutral_PU, s=pt_Neutral_PU_puppi * 20.0, c='orange',
               label='PU Neu')
    ax.scatter(eta_Neutral_LV, phi_Neutral_LV, s=pt_Neutral_LV_puppi * 20.0, c='blue',
               label='LV Neu')
    ax.scatter(eta_Charge_PU, phi_Charge_PU, s=pt_Charge_PU_puppi * 20.0, c='red',
               label='PU Chg')
    ax.scatter(eta_Charge_LV, phi_Charge_LV, s=pt_Charge_LV_puppi * 20.0, c='green',
               label='LV Chg')
    # ax.scatter(eta_boost, phi_boost, s=pt_diff * 20.0, c='green',
    #          label='effective Boost')
    ax.set_ylabel(r'$\phi$')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r'$\eta$')
    ax.set_xlim(-4, 4)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0.)
    plt.savefig(dir_name + "mitigated_pt_puppi_" + postfix + ".pdf")
    plt.close()

    if hybrid_bol == True:
        # plot with boost
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(eta_Neutral_PU, phi_Neutral_PU, s=pt_Neutral_PU_withboost * 20.0, c='orange',
                   label='PU Neu')
        ax.scatter(eta_Neutral_LV, phi_Neutral_LV, s=pt_Neutral_LV_withboost * 20.0, c='blue',
                   label='LV Neu')
        ax.scatter(eta_Charge_PU, phi_Charge_PU, s=pt_Charge_PU_withboost * 20.0, c='red',
                   label='PU Chg')
        ax.scatter(eta_Charge_LV, phi_Charge_LV, s=pt_Charge_LV_withboost * 20.0, c='green',
                   label='LV Chg')
        ax.set_ylabel(r'$\phi$')
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(r'$\eta$')
        ax.set_xlim(-4, 4)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0.)
        plt.savefig(dir_name + "mitigated_pt_withboost_" + postfix + ".pdf")
        plt.close()

        # plot the difference
        """
        pt_Neutral_PU_diff = pt_Neutral_PU_withboost - pt_Neutral_PU_withpred
        pt_Neutral_LV_diff = pt_Neutral_LV_withboost - pt_Neutral_LV_withpred
        pt_Charge_PU_diff = pt_Charge_PU_withboost - pt_Charge_PU_withpred
        pt_Charge_LV_diff = pt_Charge_LV_withboost - pt_Charge_LV_withpred
        np.round(pt_Neutral_PU_diff, 3)
        np.round(pt_Neutral_LV_diff, 3)
        np.round(pt_Charge_PU_diff, 3)
        np.round(pt_Charge_LV_diff, 3)

        sig_level = 0.1
        print(pt_Neutral_PU_diff[abs(pt_Neutral_PU_diff) > sig_level].shape[0])
        print(pt_Neutral_LV_diff[abs(pt_Neutral_LV_diff) > sig_level].shape[0])
        print(pt_Charge_PU_diff[abs(pt_Charge_PU_diff) > sig_level].shape[0])
        print(pt_Charge_LV_diff[abs(pt_Charge_LV_diff) > sig_level].shape[0])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(eta_Neutral_PU[abs(pt_Neutral_PU_diff) > sig_level], phi_Neutral_PU[abs(pt_Neutral_PU_diff) > sig_level],
                   s=pt_Neutral_PU_diff[abs(pt_Neutral_PU_diff) > sig_level] * 20.0, c='orange',
                   label='PU Neu')
        for i in range(pt_Neutral_PU_diff[abs(pt_Neutral_PU_diff) > sig_level].shape[0]):
            ax.annotate(pt_Neutral_PU_diff[abs(pt_Neutral_PU_diff) > sig_level][i], (
            eta_Neutral_PU[abs(pt_Neutral_PU_diff) > sig_level][i], phi_Neutral_PU[abs(pt_Neutral_PU_diff) > sig_level][i]),
                        fontsize=10)

        ax.scatter(eta_Neutral_LV[abs(pt_Neutral_LV_diff) > sig_level], phi_Neutral_LV[abs(pt_Neutral_LV_diff) > sig_level],
                   s=pt_Neutral_LV_diff[abs(pt_Neutral_LV_diff) > sig_level] * 20.0, c='blue',
                   label='LV Neu')
        for i in range(pt_Neutral_LV_diff[abs(pt_Neutral_LV_diff) > sig_level].shape[0]):
            ax.annotate(pt_Neutral_LV_diff[abs(pt_Neutral_LV_diff) > sig_level][i], (
            eta_Neutral_LV[abs(pt_Neutral_LV_diff) > sig_level][i], phi_Neutral_LV[abs(pt_Neutral_LV_diff) > sig_level][i]),
                        fontsize=10)

        ax.scatter(eta_Charge_PU[abs(pt_Charge_PU_diff) > sig_level], phi_Charge_PU[abs(pt_Charge_PU_diff) > sig_level],
                   s=pt_Charge_PU_diff[abs(pt_Charge_PU_diff) > sig_level] * 20.0, c='red',
                   label='PU Chg')
        for i in range(pt_Charge_PU_diff[abs(pt_Charge_PU_diff) > sig_level].shape[0]):
            ax.annotate(pt_Charge_PU_diff[abs(pt_Charge_PU_diff) > sig_level][i], (
            eta_Charge_PU[abs(pt_Charge_PU_diff) > sig_level][i], phi_Charge_PU[abs(pt_Charge_PU_diff) > sig_level][i]),
                        fontsize=10)

        ax.scatter(eta_Charge_LV[abs(pt_Charge_LV_diff) > sig_level], phi_Charge_LV[abs(pt_Charge_LV_diff) > sig_level],
                   s=pt_Charge_LV_diff[abs(pt_Charge_LV_diff) > sig_level] * 20.0, c='green',
                   label='LV Chg')
        for i in range(pt_Charge_LV_diff[abs(pt_Charge_LV_diff) > sig_level].shape[0]):
            ax.annotate(pt_Charge_LV_diff[abs(pt_Charge_LV_diff) > sig_level][i], (
                eta_Charge_LV[abs(pt_Charge_LV_diff) > sig_level][i], phi_Charge_LV[abs(pt_Charge_LV_diff) > sig_level][i]),
                        fontsize=10)

        ax.set_ylabel(r'$\phi$')
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(r'$\eta$')
        ax.set_xlim(-4, 4)
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0.)
        filename = dir_name + "diff_boost_pred_" + postfix + ".pdf"
        plt.savefig(filename)
        plt.close()
        """

def plot_boost_train(epochs_test, loss_graph_train, loss_graph_train_hybrid,
                         auc_graph_train_puppi, auc_graph_train_puppi_neu,
                         auc_graph_train, auc_graph_neu_train,
                         auc_graph_train_hybrid, auc_graph_neu_train_hybrid, postfix=".pdf"):
    # loss
    plt.figure()
    # plt.plot(epochs_train, loss_graph, label='train', linestyle='solid', linewidth=1, color='r')
    plt.plot(epochs_test, loss_graph_train, label='GNN prediction', linestyle='solid', linewidth=1, color='g')
    plt.plot(epochs_test, loss_graph_train_hybrid, label='hybrid algorithm', linestyle='solid', linewidth=1, color='b')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend(loc=4)
    plt.savefig(dir_name + "loss_graph_hybrid" + postfix)
    plt.close()

    # auc
    plt.figure()
    plt.plot(epochs_test, auc_graph_train, label="GNN Charge", linestyle='dashed', linewidth=1, color='r')
    plt.plot(epochs_test, auc_graph_train_hybrid, label="Hybrid Charge", linestyle='solid', linewidth=1, color='b')
    plt.plot(epochs_test, auc_graph_train_puppi, label="PUPPI Charge", linestyle='solid', linewidth=1,
             color='g')
    plt.plot(epochs_test, auc_graph_neu_train, label="GNN Neutral", linestyle='dashed', linewidth=1,
             color='orange')
    plt.plot(epochs_test, auc_graph_neu_train_hybrid, label="Hybrid Neutral", linestyle='solid', linewidth=1, color='cyan')
    plt.plot(epochs_test, auc_graph_train_puppi_neu, label="PUPPI Neu train", linestyle='solid',
             linewidth=1, color='m')
    plt.xlabel('Epochs')
    plt.ylabel('auc')
    plt.legend(loc=4)
    plt.savefig(dir_name + "auc_graph_hybrid" + postfix)
    plt.close()
