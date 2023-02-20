"""
script to test the physics performance of a given model
"""

import scipy.stats
from collections import OrderedDict
from pyjet import cluster, DTYPE_PTEPM
import argparse
import torch
from torch_geometric.data import DataLoader
import models.models as models
import utils.utils
import matplotlib
from copy import deepcopy
import os
import copy
import uproot
import awkward as ak

# matplotlib.use("pdf")
import numpy as np
import random
import pickle
import joblib
from timeit import default_timer as timer
from tqdm import tqdm

import matplotlib as mpl
import imageio

# mpl.use("pdf")
import matplotlib.pyplot as plt
import mplhep as hep

hep.set_style(hep.style.CMS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def deltaPhiNew(dphis):
    dphis = np.where(dphis > np.pi, dphis - 2*np.pi, dphis)
    dphis = np.where(dphis < -np.pi, dphis + 2*np.pi, dphis)
    return dphis


def deltaRNew(detas, dphis):
    """
    calculate the deltaR based on the input deta and phi
    """
    dphis = deltaPhiNew(dphis)
    dR = np.sqrt(detas**2 + dphis**2)
    return dR


def generate_mask(dataset):
    # how many LV and PU to sample
    # dataset = deepcopy(dataset_org)
    for graph in dataset:
        LV_index = graph.LV_index
        PU_index = graph.PU_index
        original_feature = graph.x[:, 0:graph.num_feature_actual]

        num_select_LV = 5
        num_select_PU = 50

        if LV_index.shape[0] < num_select_LV or PU_index.shape[0] < num_select_PU:
            num_select_LV = min(LV_index.shape[0], num_select_LV)
            num_select_PU = min(PU_index.shape[0], num_select_PU)

        # generate the index for LV and PU samples for training mask
        gen_index_LV = random.sample(range(LV_index.shape[0]), num_select_LV)
        selected_LV_train = LV_index[gen_index_LV]

        gen_index_PU = random.sample(range(PU_index.shape[0]), num_select_PU)
        selected_PU_train = PU_index[gen_index_PU]

        training_mask = np.concatenate(
            (selected_LV_train, selected_PU_train), axis=None)
        # print(training_mask)

        # construct mask vector for training and testing
        mask_training = torch.zeros(graph.num_nodes, 1)
        mask_training[[training_mask.tolist()]] = 1

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

        # pf_dz_training_test=torch.clone(original_feature[:,6:7])
        # pf_dz_training_test = torch.zeros(graph.num_nodes, 1)

        # -4 is for one hot encoding of fromLV and one mask; -1 is for final puppiweight
        # default_data_training = torch.cat(
        #   (original_feature[:, 0:(graph.num_features - 4 - 1)], puppiWeight_default_one_hot_training,
        #    original_feature[:, -1].view(-1, 1)), 1)
        # default_data_training = torch.cat(
        #     (original_feature[:, 0:(graph.num_feature_actual - 7)],pdgId_one_hot_training, pf_dz_training_test ,puppiWeight_default_one_hot_training), 1)
        default_data_training = torch.cat(
            (original_feature[:, 0:(graph.num_feature_actual - 6)], pdgId_one_hot_training, puppiWeight_default_one_hot_training), 1)

        concat_default = torch.cat((graph.x, default_data_training), 1)
        graph.x = concat_default


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


class Args(object):
    """
    arguments for loading models
    """

    def __init__(self, model_type='Gated', do_boost=False, extralayers=False):
        self.model_type = model_type
        self.num_layers = 2
        self.batch_size = 1
        self.hidden_dim = 20
        self.dropout = 0
        self.opt = 'adam'
        self.weight_decay = 0
        self.lr = 0.01
        self.do_boost = do_boost
        self.extralayers = extralayers


class PerformanceMetrics(object):
    """
    physics performance metrics
    """

    def __init__(self):
        pt = 0.
        pt_diff = 0.
        mass_diff = 0.
        dR_diff = 0.


def clusterJets(pt, eta, phi, ptcut=0., deltaR=0.4):
    """
    cluster the jets based on the array of pt, eta, phi,
    of all particles (masses are assumed to be zero),
    with pyjet clustering algo
    """
    # cleaning zero pt-ed objects
    pt_wptcut = pt[pt > ptcut]
    eta_wptcut = eta[pt > ptcut]
    phi_wptcut = phi[pt > ptcut]
    mass_wptcut = np.zeros(pt_wptcut.shape[0])

    event = np.column_stack((pt_wptcut, eta_wptcut, phi_wptcut, mass_wptcut))
    event.dtype = DTYPE_PTEPM
    sequence = cluster(event, R=deltaR, p=-1)
    jets = sequence.inclusive_jets(ptmin=30)

    return jets


def deltaPhi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return dphi


def deltaR(eta1, phi1, eta2, phi2):
    """
    calculate the deltaR between two jets/particles
    """
    deta = eta1 - eta2
    dphi = phi1 - phi2
    while dphi > np.pi:
        dphi -= 2 * np.pi
    while dphi < -np.pi:
        dphi += 2 * np.pi
    return np.hypot(deta, dphi)


def deltaRJet(jet1, jet2):
    """
    calculate the deltaR of the two PseudoJet
    """
    return deltaR(jet1.eta, jet1.phi, jet2.eta, jet2.phi)


def matchJets(jets_truth, jets_reco, dRcut=0.1):
    """
    match the jets in jets_reco to jets_truth,
    based on the deltaR
    """
    matched_indices = []

    jets_truth_indices = list(range(len(jets_truth)))
    jets_reco_indices = list(range(len(jets_reco)))

    for ijet_reco in jets_reco_indices:
        for ijet_truth in jets_truth_indices:
            # print("deltR between {} and {} is {}".format(ijet_truth, ijet_reco, deltaRJet(jets_truth[ijet_truth], jets_reco[ijet_reco])))
            if deltaRJet(jets_truth[ijet_truth], jets_reco[ijet_reco]) < dRcut:
                matched_indices.append((ijet_truth, ijet_reco))
                jets_truth_indices.remove(ijet_truth)
                break

    return matched_indices


def compareJets(jets_truth, jets_reco, dRcut=0.1):
    """
    match jets between truth and reco using matchJets,
    and then compared the matched deltaR, pt, and mass
    """
    performances = []

    matched_indices = matchJets(jets_truth, jets_reco, dRcut=dRcut)
    for ijet_truth, ijet_reco in matched_indices:
        perf = PerformanceMetrics()
        perf.pt_truth = jets_truth[ijet_truth].pt
        perf.mass_diff = (
            jets_reco[ijet_reco].mass - jets_truth[ijet_truth].mass)/(jets_truth[ijet_truth].mass+1e-6)
        perf.pt_diff = (
            jets_reco[ijet_reco].pt - jets_truth[ijet_truth].pt)/(jets_truth[ijet_truth].pt+1e-6)
        perf.dR_diff = deltaRJet(jets_truth[ijet_truth], jets_reco[ijet_reco])
        performances.append(perf)
    return performances


def calculateMET(pt, phi):
    """
    calculate the MET based on all particles pt and phi
    """
    met_x = np.sum(pt * np.cos(phi), axis=0)
    met_y = np.sum(pt * np.sin(phi), axis=0)
    return np.hypot(met_x, met_y)


def postProcessing(data, preds):
    """
    reconstruct jet and MET,
    compare the reco-ed jet and MET with truth ones,
    using the input data and ML weights (pred)
    """
    pt = np.array(data.x[:, 2].cpu().detach())
    eta = np.array(data.x[:, 0].cpu().detach())
    phi = np.array(data.x[:, 1].cpu().detach())
    puppi = np.array(data.pWeight.cpu().detach())
    # puppi = np.array(data.x[:,data.num_feature_actual[0].item()-1].cpu().detach())
    # truth = np.array(data.y.cpu().detach())

    pt_truth = np.array(data.GenPart_nump[:, 2].cpu().detach())
    eta_truth = np.array(data.GenPart_nump[:, 0].cpu().detach())
    phi_truth = np.array(data.GenPart_nump[:, 1].cpu().detach())
    # print (pt)

    # print(truth)
    # pred = np.array(pred[:,0].cpu().detach())
    # pred2 = np.array(pred2[:, 0].cpu().detach())
    # set all particle masses to zero
    mass = np.zeros(pt.shape[0])

    # remove pt < 0.5 particles
    pt[pt < 0.5] = 0

    # apply CHS to puppi weights
    charge_index = data.Charge_index[0]
    # puppi[charge_index] = truth[charge_index]
    # apply CHS to predicted weights
    # pred[charge_index] = puppi[charge_index]
    # pred2[charge_index] = truth[charge_index]

    # truth information
    # pt_truth   = pt * truth

    # puppi information
    pt_puppi = pt * puppi
    # apply some weight cuts on puppi
    cut = 0.41  # GeV
    wcut = 0.17
    cut = 0.0
    wcut = 0.0
    # cut = 0.99 #GeV
    # cut = 0.99 #GeV
    # wcut = 0.15
    # cut = 1.242 #GeV
    # wcut = 0.115
    pt_puppi_wcut = np.array(pt, copy=True)
    pt_puppi_wcut[(puppi < wcut) | (pt_puppi < cut)] = 0.
    # apply CHS
    # pt_puppi_wcut[charge_index] = pt_puppi[charge_index]

    # prediction information
    # pt_pred = pt * pred
    # pt_pred2 = pt * pred2

    # cluster jets with truth particles
    jets_truth = clusterJets(pt_truth, eta_truth, phi_truth)
    # print (jets_truth)

    jets_puppi = clusterJets(pt_puppi, eta, phi)
    performances_jet_puppi = compareJets(jets_truth, jets_puppi)

    jets_puppi_wcut = clusterJets(pt_puppi_wcut, eta, phi)
    performances_jet_puppi_wcut = compareJets(jets_truth, jets_puppi_wcut)

    # jets_pred  = clusterJets(pt_pred,  eta, phi)
    # print("pt_pred: ", jets_pred)
    # performances_jet_pred = compareJets(jets_truth, jets_pred)

    # jets_pred2 = clusterJets(pt_pred2, eta, phi)
    # print("pt_pred2", jets_pred2)
    # performances_jet_pred2 = compareJets(jets_truth, jets_pred2)

    # calculate MET and compare
    met_truth = calculateMET(pt_truth, phi_truth)
    met_puppi = calculateMET(pt_puppi, phi)
    met_puppi_wcut = calculateMET(pt_puppi_wcut, phi)
    # met_pred  = calculateMET(pt_pred,  phi)
    # met_pred2 = calculateMET(pt_pred2, phi)
    # print("***** one event ********")
    # print("met truth", met_truth)
    # print("met puppi", met_puppi)
    # print("met puppi wcut", met_puppi_wcut)
    # print("met pred", met_pred)

    # evaluate the performances for the predictions
    performances_jet_pred = []
    mets_pred = []

    for pred in preds:
        # print("preds: ", pred)
        pred = np.array(pred[0][:, 0].cpu().detach())

        # apply CHS to predictions
        # charge_index = data.Charge_index[0]
        pred[charge_index] = puppi[charge_index]
        pt_pred = pt * pred
        jets_pred = clusterJets(pt_pred,  eta, phi)
        performance_jet_pred = compareJets(jets_truth, jets_pred)

        # MET
        met_pred = calculateMET(pt_pred,  phi)

        performances_jet_pred.append(performance_jet_pred)
        mets_pred.append(met_pred)

    return met_truth, performances_jet_puppi, met_puppi, performances_jet_puppi_wcut, met_puppi_wcut, performances_jet_pred, mets_pred


def test(filelists, models={}):

    for model in models.values():
        model.to('cuda:0')
        model.eval()

    performances_jet_puppi = []
    performances_jet_puppi_wcut = []

    mets_truth = []
    mets_puppi = []
    mets_puppi_wcut = []

    performances_jet_pred = OrderedDict()
    mets_pred = OrderedDict()
    for modelname in models.keys():
        performances_jet_pred[modelname] = []
        mets_pred[modelname] = []

    ievt = 0
    for ifile in filelists:
        print("ifile: ", ifile)
        fp = open(ifile, "rb")
        dataset = joblib.load(fp)
        generate_neu_mask(dataset)
        generate_mask(dataset)
        data = DataLoader(dataset, batch_size=1)
        loader = data

        for data in loader:
            ievt += 1
            # if ievt > 10:
            #    break

            if ievt % 10 == 0:
                print("processed {} events".format(ievt))
            with torch.no_grad():
                data = data.to(device)
                # max(dim=1) returns values, indices tuple; only need indices

                # loop over model in models and run the inference
                preds = []

                for model in models.values():
                    model.to('cuda:0')
                    model.eval()

                    pred = model.forward(data)
                    # print("pred here: ", pred)
                    preds.append(pred)

                met_truth, perfs_jet_puppi, met_puppi, perfs_jet_puppi_wcut, met_puppi_wcut, perfs_jet_pred, mets_fromF_pred = postProcessing(
                    data, preds)
                # perfs_jet_puppi, perfs_jet_puppi_wcut, perfs_jet_pred, perfs_jet_pred2, met_truth, met_puppi, met_puppi_wcut, met_pred, met_pred2 = postProcessing(data, preds)

                performances_jet_puppi += perfs_jet_puppi
                performances_jet_puppi_wcut += perfs_jet_puppi_wcut
                # performances_jet_pred += perfs_jet_pred
                # performances_jet_pred2 += perfs_jet_pred2

                mets_truth.append(met_truth)
                mets_puppi.append(met_puppi)
                mets_puppi_wcut.append(met_puppi_wcut)

                imodel = 0
                for modelname in models.keys():
                    # print("modelname ", perfs_jet_pred[imodel])
                    # print("performances_jet_pred modelname", performances_jet_pred[modelname] )
                    performances_jet_pred[modelname] += perfs_jet_pred[imodel]
                    mets_pred[modelname].append(mets_fromF_pred[imodel])
                    imodel += 1

        fp.close()

    return mets_truth, performances_jet_puppi, mets_puppi, performances_jet_puppi_wcut, mets_puppi_wcut, performances_jet_pred, mets_pred


def main(modelname, filelists):
    # load models
    args = Args()
    model_gated_boost = models.GNNStack(9, args.hidden_dim, 1, args)
    # model_load.load_state_dict(torch.load('best_valid_model_semi.pt'))
    model_gated_boost.load_state_dict(torch.load(modelname))

    modelcolls = OrderedDict()
    modelcolls['gated_boost'] = model_gated_boost

    # run the tests
    filelists = ["/Workdir/data_pickle/dataset_graph_puppi_8000"]
    mets_truth, performances_jet_puppi, mets_puppi, performances_jet_puppi_wcut, mets_puppi_wcut, performances_jet_pred, mets_pred = test(
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

    %matplotlib inline
    plt.style.use(hep.style.ROOT)
    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_pred0])
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='blue', linewidth=linewidth,
             density=False, label=r'Semi-supervised, $\mu={:10.2f}$, $\sigma={:10.2f}$'.format(*(getStat(mass_diff))))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_puppi])
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='green', linewidth=linewidth,
             density=False, label=r'PUPPI, $\mu={:10.2f}$, $\sigma={:10.2f}$'.format(*(getStat(mass_diff))))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_puppi_wcut])
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='red', linewidth=linewidth,
             density=False, label=r'PF, $\mu={:10.2f}$, $\sigma={:10.2f}$'.format(*(getStat(mass_diff))))
    # plt.xlim(-1.0,1.3)
    plt.xlabel(r"Jet Mass $(m_{reco} - m_{truth})/m_{truth}$")
    plt.ylabel('A.U.')
    plt.ylim(0, 2000)
    plt.legend()
    plt.savefig("Jet_mass_diff.pdf")
    plt.show()

    # %matplotlib inline
    fig = plt.figure(figsize=(10, 8))

    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_pred0])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='blue', linewidth=linewidth,
             density=False, label=r'Semi-supevised, $\mu={:10.3f}$, $\sigma={:10.3f}$'.format(*(getStat(pt_diff))))
    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_puppi])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='green', linewidth=linewidth,
             density=False, label=r'PUPPI, $\mu={:10.3f}$, $\sigma={:10.3f}$'.format(*(getStat(pt_diff))))
    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_puppi_wcut])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='red', linewidth=linewidth,
             density=False, label=r'PF, $\mu={:10.3f}$, $\sigma={:10.3f}$'.format(*(getStat(pt_diff))))
    # plt.xlim(0,40)
    plt.ylim(0, 2000)
    plt.xlabel(r"Jet $p_{T}$ $(p^{reco}_{T} - p^{truth}_{T})/p^{truth}_{T}$")
    plt.ylabel('A.U.')
    plt.legend()
    plt.show()
    plt.savefig("Jet_pT_diff.pdf")

    # MET resolution
    # %matplotlib inline

    fig = plt.figure(figsize=(10, 8))
    mets_diff = (np.array(mets_pred0) - np.array(mets_truth))
    plt.hist(mets_diff, bins=30, range=(-30, 30), histtype='step', color='blue', linewidth=linewidth,
             density=False, label=r'Semi-supervised, $\mu={:10.2f}$, $\sigma={:10.2f}$'.format(*(getStat(mets_diff))))
    mets_diff = (np.array(mets_puppi) - np.array(mets_truth))
    plt.hist(mets_diff, bins=30, range=(-30, 30), histtype='step', color='green', linewidth=linewidth,
             density=False, label=r'PUPPI, $\mu={:10.2f}$, $\sigma={:10.2f}$'.format(*(getStat(mets_diff))))
    mets_diff = (np.array(mets_puppi_wcut) - np.array(mets_truth))
    plt.hist(mets_diff, bins=30, range=(-30, 30), histtype='step', color='red', linewidth=linewidth,
             density=False, label=r'PF, $\mu={:10.2f}$, $\sigma={:10.2f}$'.format(*(getStat(mets_diff))))

    plt.xlabel(r"$p^{miss, reco}_{T} - p^{miss, truth}_{T}$ [GeV]")
    plt.ylabel('A.U.')
    plt.ylim(0, 500)
    plt.legend()
    plt.show()
    plt.savefig("MET_diff.pdf")

    # more plots to be included


if __name__ == '__main__':
    modelname = "/Workdir/fast_simulation/test/best_valid_model.pt"
    filelists = ["/Workdir/data_pickle/dataset_graph_puppi_8000"]
    main(modelname, filelists)
