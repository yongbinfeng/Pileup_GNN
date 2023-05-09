"""
script to test the physics performance of a given model
"""

import scipy.stats
from collections import OrderedDict
from pyjet import cluster, DTYPE_PTEPM
import argparse
import torch
from torch_geometric.data import DataLoader
import models as models
import utils
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
testneu = 1
# Options to Chg+Neu or Chg only


def NormaliseDeltaPhi(dphis):
    dphis = np.where(dphis > np.pi, dphis - 2*np.pi, dphis)
    dphis = np.where(dphis < -np.pi, dphis + 2*np.pi, dphis)
    return dphis


def NormaliseDeltaRNew(detas, dphis):
    """
    calculate the deltaR based on the input deta and phi
    """
    dphis = NormaliseDeltaPhi(dphis)
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
        self.num_layers = 3
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


def clusterJets(pt, eta, phi, ifAK8, ptcut=0., deltaR=0.4):
    """
    cluster the jets based on the array of pt, eta, phi,
    of all particles (masses are assumed to be zero),
    with pyjet clustering algo
    """
    # cleaning zero pt-ed objects
    if ifAK8==1:
      deltaR=0.8
    pt_wptcut = pt[pt > ptcut]
    eta_wptcut = eta[pt > ptcut]
    phi_wptcut = phi[pt > ptcut]
    mass_wptcut = np.zeros(pt_wptcut.shape[0])

    event = np.column_stack((pt_wptcut, eta_wptcut, phi_wptcut, mass_wptcut))
    event.dtype = DTYPE_PTEPM
    sequence = cluster(event, R=deltaR, p=-1)
    Ptmin = 30
    if ifAK8:
        Ptmin = 300
    jets = sequence.inclusive_jets(ptmin=Ptmin)
    #charged only
    #jets = sequence.inclusive_jets(ptmin=20)

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
        perf.mass_truth = jets_truth[ijet_truth].mass
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

def ExtractJet(jets):
    
    ptlist = []
    etalist = []
    philist = []
    masslist = []
    njets = len(jets)
    for i in range(0, njets):
        ptlist.append(jets[i].pt)
        etalist.append(jets[i].eta)
        philist.append(jets[i].phi)
        masslist.append(jets[i].mass)
    
    return njets, ptlist, etalist, philist, masslist


def postProcessing(data, preds,ifAK8):
    """
    reconstruct jet and MET,
    compare the reco-ed jet and MET with truth ones,
    using the input data and ML weights (pred)
    """
    pt = np.array(data.x[:, 2].cpu().detach())
    eta = np.array(data.x[:, 0].cpu().detach())
    phi = np.array(data.x[:, 1].cpu().detach())
    puppi = np.array(data.pWeight.cpu().detach())
    puppichg = np.array(data.pWeightchg.cpu().detach())
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
    pt[pt < 0.01] = 0

    # apply CHS to puppi weights
    charge_index = data.Charge_index[0]
    neutral_index = data.Neutral_index[0]
    Gencharge = np.array(data.GenPart_nump[:,4].cpu().detach())
    
    lv_index = data.LV_index[0]
    pu_index = data.PU_index[0]

    chglv_index = list(set(lv_index) & set(charge_index))
    chgpu_index = list(set(pu_index) & set(charge_index))
    # puppi[charge_index] = truth[charge_index]
    # apply CHS to predicted weights
    # pred[charge_index] = puppi[charge_index]
    # pred2[charge_index] = truth[charge_index]

    # truth information
    # pt_truth   = pt * truth

    
    if testneu == 1:
        chargeOnly = 0
    else:
        chargeOnly = 1

    if chargeOnly == 1 :
       pt[neutral_index] = 0
       pt_truth[Gencharge==0] = 0

    # puppi information
    if testneu == 1:
        puppichg[charge_index] = puppi[charge_index]
    if testneu == 0:
        pt_puppi = pt * puppichg
    if testneu == 1:
        pt_puppi = pt * puppi
    pt_CHS = pt * puppi
    if testneu == 1:
        pt_CHS[neutral_index] = pt[neutral_index]
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
    jets_truth = clusterJets(pt_truth, eta_truth, phi_truth, ifAK8)
    # print (jets_truth)
    njets_truth, pt_jets_truth, eta_jets_truth, phi_jets_truth, mass_jets_truth = ExtractJet(jets_truth)

    
    jets_puppi = clusterJets(pt_puppi, eta, phi, ifAK8)
    njets_puppi, pt_jets_puppi, eta_jets_puppi, phi_jets_puppi, mass_jets_puppi = ExtractJet(jets_puppi)
    performances_jet_puppi = compareJets(jets_truth, jets_puppi)

    jets_puppi_wcut = clusterJets(pt_puppi_wcut, eta, phi, ifAK8)
    njets_pf, pt_jets_pf, eta_jets_pf, phi_jets_pf, mass_jets_pf = ExtractJet(jets_puppi_wcut)
    performances_jet_puppi_wcut = compareJets(jets_truth, jets_puppi_wcut)

    jets_CHS = clusterJets(pt_CHS, eta, phi, ifAK8)
    njets_CHS, pt_jets_CHS, eta_jets_CHS, phi_jets_CHS, mass_jets_CHS = ExtractJet(jets_CHS)
    performances_jet_CHS = compareJets(jets_truth, jets_CHS)

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
    neu_pred = []
    neu_puppi = []
    chlv_pred = []
    chpu_pred = []
    chlv_puppi = []
    chpu_puppi = []

    for pred in preds:
        # print("preds: ", pred)
        pred = np.array(pred[0][:, 0].cpu().detach())
        #pred[pred<0.3] = 0
        predcopy = pred
        predcopyA = []
        predcopyB = []
        for j in range(len(predcopy)):
            predcopyA.append(predcopy[j])

        predcopy[charge_index] = -2
        for m in range(len(predcopy)):
            if predcopy[m]>-0.1:
                neu_pred.append(predcopy[m])
                neu_puppi.append(puppichg[m])
        
        predcopyA = np.array(predcopyA)
        #predcopyA[predcopyA<0.3] = 0
        #predcopyA[predcopyA>0.3] = 1


        for mi in chglv_index:
            chlv_pred.append(predcopyA[mi])
            chlv_puppi.append(puppichg[mi])
        for mj in chgpu_index:
            chpu_pred.append(predcopyA[mj])
            chpu_puppi.append(puppichg[mj])
        # apply CHS to predictions
        # charge_index = data.Charge_index[0]
        #pred[charge_index] = puppichg[charge_index]
        if testneu == 1:
            predcopyA[charge_index] = puppi[charge_index]
        
        pt_pred = pt * predcopyA
        jets_pred = clusterJets(pt_pred,  eta, phi, ifAK8)
        njets_pred, pt_jets_pred, eta_jets_pred, phi_jets_pred, mass_jets_pred = ExtractJet(jets_pred)
        performance_jet_pred = compareJets(jets_truth, jets_pred)

        # MET
        met_pred = calculateMET(pt_pred,  phi)

        performances_jet_pred.append(performance_jet_pred)
        mets_pred.append(met_pred)
    
    

    return met_truth,performances_jet_CHS, performances_jet_puppi, met_puppi, performances_jet_puppi_wcut, met_puppi_wcut, performances_jet_pred, mets_pred, neu_pred, neu_puppi, chlv_pred, chpu_pred, chlv_puppi, chpu_puppi, njets_pf, njets_pred, njets_puppi, njets_truth, njets_CHS, pt_jets_pf, pt_jets_pred, pt_jets_puppi, pt_jets_truth, pt_jets_CHS, eta_jets_pf, eta_jets_pred, eta_jets_puppi, eta_jets_truth, eta_jets_CHS, phi_jets_pf, phi_jets_pred, phi_jets_puppi, phi_jets_truth, phi_jets_CHS, mass_jets_pf, mass_jets_pred, mass_jets_puppi, mass_jets_truth, mass_jets_CHS


def test(filelists, models={}):

    for model in models.values():
        model.to('cuda:0')
        model.eval()

    performances_jet_puppi = []
    performances_jet_CHS = []
    performances_jet_puppi_wcut = []

    mets_truth = []
    mets_puppi = []
    mets_puppi_wcut = []

    neu_weight = []
    neu_puppiweight = []
    chlv_weight = []
    chpu_weight = []
    chlv_puppiweight = []
    chpu_puppiweight = []
    pt_jets_pf = []
    pt_jets_pred = []
    pt_jets_puppi = []
    pt_jets_truth = []
    pt_jets_CHS = []

    eta_jets_pf = []
    eta_jets_pred = []
    eta_jets_puppi = []
    eta_jets_truth = []
    eta_jets_CHS = []

    phi_jets_pf = []
    phi_jets_pred = []
    phi_jets_puppi = []
    phi_jets_truth = []
    phi_jets_CHS = []

    mass_jets_pf = []
    mass_jets_pred = []
    mass_jets_puppi = []
    mass_jets_truth = []
    mass_jets_CHS = []

    njets_pf = []
    njets_pred = []
    njets_puppi = []
    njets_truth = []
    njets_CHS = []

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

        ifAK8 = 0
        if (ifile=="data_pickle/dataset_graph_puppi_test_Wjets4000"):
            ifAK8 = 1
        

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

                met_truth, perfs_jet_CHS, perfs_jet_puppi, met_puppi, perfs_jet_puppi_wcut, met_puppi_wcut, perfs_jet_pred, mets_fromF_pred, neus_pred, neus_puppi, chlvs_pred, chpus_pred, chlvs_puppi, chpus_puppi, Njets_pf, Njets_pred, Njets_puppi, Njets_truth, Njets_CHS, Pt_jets_pf, Pt_jets_pred, Pt_jets_puppi, Pt_jets_truth, Pt_jets_CHS, Eta_jets_pf, Eta_jets_pred, Eta_jets_puppi, Eta_jets_truth, Eta_jets_CHS, Phi_jets_pf, Phi_jets_pred, Phi_jets_puppi, Phi_jets_truth, Phi_jets_CHS, Mass_jets_pf, Mass_jets_pred, Mass_jets_puppi, Mass_jets_truth, Mass_jets_CHS = postProcessing(
                    data, preds, ifAK8)
                # perfs_jet_puppi, perfs_jet_puppi_wcut, perfs_jet_pred, perfs_jet_pred2, met_truth, met_puppi, met_puppi_wcut, met_pred, met_pred2 = postProcessing(data, preds)

                performances_jet_puppi += perfs_jet_puppi
                performances_jet_CHS += perfs_jet_CHS
                performances_jet_puppi_wcut += perfs_jet_puppi_wcut
                # performances_jet_pred += perfs_jet_pred
                # performances_jet_pred2 += perfs_jet_pred2

                mets_truth.append(met_truth)
                mets_puppi.append(met_puppi)
                mets_puppi_wcut.append(met_puppi_wcut)

                imodel = 0

                njets_pf.append(Njets_pf)
                njets_puppi.append(Njets_puppi)
                njets_pred.append(Njets_pred)
                njets_truth.append(Njets_truth)
                njets_CHS.append(Njets_CHS)

                for ipf in range(0, Njets_pf):
                    pt_jets_pf.append(Pt_jets_pf[ipf])
                    eta_jets_pf.append(Eta_jets_pf[ipf])
                    phi_jets_pf.append(Phi_jets_pf[ipf])
                    mass_jets_pf.append(Mass_jets_pf[ipf])

                for ipuppi in range(0, Njets_puppi):
                    pt_jets_puppi.append(Pt_jets_puppi[ipuppi])
                    eta_jets_puppi.append(Eta_jets_puppi[ipuppi])
                    phi_jets_puppi.append(Phi_jets_puppi[ipuppi])
                    mass_jets_puppi.append(Mass_jets_puppi[ipuppi])

                for ipred in range(0, Njets_pred):
                    pt_jets_pred.append(Pt_jets_pred[ipred])
                    eta_jets_pred.append(Eta_jets_pred[ipred])
                    phi_jets_pred.append(Phi_jets_pred[ipred])
                    mass_jets_pred.append(Mass_jets_pred[ipred])

                for itruth in range(0, Njets_truth):
                    pt_jets_truth.append(Pt_jets_truth[itruth])
                    eta_jets_truth.append(Eta_jets_truth[itruth])
                    phi_jets_truth.append(Phi_jets_truth[itruth])
                    mass_jets_truth.append(Mass_jets_truth[itruth])

                for ic in range(0, Njets_CHS):
                    pt_jets_CHS.append(Pt_jets_CHS[ic])
                    eta_jets_CHS.append(Eta_jets_CHS[ic])
                    phi_jets_CHS.append(Phi_jets_CHS[ic])
                    mass_jets_CHS.append(Mass_jets_CHS[ic])


                for m0 in range(len(neus_pred)):
                    neu_puppiweight.append(neus_puppi[m0])
                for m1 in range(len(neus_pred)):
                    neu_weight.append(neus_pred[m1])
                for m2 in range(len(chlvs_pred)):
                    chlv_weight.append(chlvs_pred[m2])
                for m3 in range(len(chpus_pred)):
                    chpu_weight.append(chpus_pred[m3])
                for m4 in range(len(chlvs_puppi)):
                    chlv_puppiweight.append(chlvs_puppi[m4])
                for m5 in range(len(chpus_puppi)):
                    chpu_puppiweight.append(chpus_puppi[m5])
                for modelname in models.keys():
                    # print("modelname ", perfs_jet_pred[imodel])
                    # print("performances_jet_pred modelname", performances_jet_pred[modelname] )
                    performances_jet_pred[modelname] += perfs_jet_pred[imodel]
                    mets_pred[modelname].append(mets_fromF_pred[imodel])
                    imodel += 1
        print("eventNum:"+str(ievt))

        fp.close()

    return mets_truth, performances_jet_CHS, performances_jet_puppi, mets_puppi, performances_jet_puppi_wcut, mets_puppi_wcut, performances_jet_pred, mets_pred, neu_weight, neu_puppiweight, chlv_weight, chpu_weight, chlv_puppiweight, chpu_puppiweight, njets_pf, njets_pred, njets_puppi, njets_truth, njets_CHS, pt_jets_pf, pt_jets_pred, pt_jets_puppi, pt_jets_truth, pt_jets_CHS, eta_jets_pf, eta_jets_pred, eta_jets_puppi, eta_jets_truth, eta_jets_CHS, phi_jets_pf, phi_jets_pred, phi_jets_puppi, phi_jets_truth, phi_jets_CHS, mass_jets_pf, mass_jets_pred, mass_jets_puppi, mass_jets_truth, mass_jets_CHS


def main(modelname, filelists):
    # load models
    args = Args()
    model_gated_boost = models.GNNStack(9, args.hidden_dim, 1, args)
    # model_load.load_state_dict(torch.load('best_valid_model_semi.pt'))
    model_gated_boost.load_state_dict(torch.load(modelname))

    modelcolls = OrderedDict()
    modelcolls['gated_boost'] = model_gated_boost

    # run the tests
    #filelists = ["../data_pickle/dataset_graph_puppi_test_40004000"]
    mets_truth, performances_jet_CHS, performances_jet_puppi, mets_puppi, performances_jet_puppi_wcut, mets_puppi_wcut, performances_jet_pred, mets_pred, neu_weight, neu_puppiweight, chlv_weight, chpu_weight, chlv_puppiweight, chpu_puppiweight, njets_pf, njets_pred, njets_puppi, njets_truth, njets_CHS, pt_jets_pf, pt_jets_pred, pt_jets_puppi, pt_jets_truth, pt_jets_CHS, eta_jets_pf, eta_jets_pred, eta_jets_puppi, eta_jets_truth, eta_jets_CHS, phi_jets_pf, phi_jets_pred, phi_jets_puppi, phi_jets_truth, phi_jets_CHS, mass_jets_pf, mass_jets_pred, mass_jets_puppi, mass_jets_truth, mass_jets_CHS = test(
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

   #  %matplotlib inline
    plt.style.use(hep.style.ROOT)
    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_pred0])
    print(mass_diff)
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='blue', linewidth=linewidth,
             density=True, label=r'Semi-supervised, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_puppi])
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='green', linewidth=linewidth, 
             density=True, label=r'PUPPI, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_puppi_wcut])
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='red', linewidth=linewidth, 
             density=True, label=r'PF, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_CHS])
    plt.hist(mass_diff, bins=40, range=(-1, 1), histtype='step', color='orange', linewidth=linewidth, 
             density=True, label=r'CHS, $\mu={:10.2f}$, $\sigma={:10.2f}$, counts:'.format(*(getStat(mass_diff)))+str(len(mass_diff)))
    # plt.xlim(-1.0,1.3)
    plt.xlabel(r"Jet Mass $(m_{reco} - m_{truth})/m_{truth}$")
    plt.ylabel('density')
    plt.ylim(0, 6)
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.savefig("Jet_mass_diff.pdf")
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    njets_pf_total = np.array(njets_pf)
    njets_pred_total = np.array(njets_pred)
    njets_puppi_total = np.array(njets_puppi)
    njets_truth_total = np.array(njets_truth)
    njets_CHS_total = np.array(njets_CHS)
    plt.hist(njets_pf_total, bins=12, range=(0, 12), histtype='step', color='blue', linewidth=linewidth,
              density=False, label=r'PF jets number')
    plt.hist(njets_pred_total, bins=12, range=(0, 12), histtype='step', color='green', linewidth=linewidth,
              density=False, label=r'SSL jets number')
    plt.hist(njets_puppi_total, bins=12, range=(0, 12), histtype='step', color='pink', linewidth=linewidth,
              density=False, label=r'puppi jets number')
    plt.hist(njets_truth_total, bins=12, range=(0, 12), histtype='step', color='black', linewidth=linewidth,
              density=False, label=r'truth jets number')
    plt.hist(njets_CHS_total, bins=12, range=(0, 12), histtype='step', color='red', linewidth=linewidth,
              density=False, label=r'CHS jets number')
    
    plt.xlabel(r"jet num")
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    plt.savefig("JetNum.pdf")

    fig = plt.figure(figsize=(10, 8))
    pt_jets_pf_total = np.array(pt_jets_pf)
    pt_jets_pred_total = np.array(pt_jets_pred)
    pt_jets_puppi_total = np.array(pt_jets_puppi)
    pt_jets_truth_total = np.array(pt_jets_truth)
    pt_jets_CHS_total = np.array(pt_jets_CHS)
    plt.hist(pt_jets_pf_total, bins=40, range=(0, 400), histtype='step', color='blue', linewidth=linewidth,
              density=False, label=r'PF jets pT')
    plt.hist(pt_jets_pred_total, bins=40, range=(0, 400), histtype='step', color='green', linewidth=linewidth,
              density=False, label=r'SSL jets pT')
    plt.hist(pt_jets_puppi_total, bins=40, range=(0, 400), histtype='step', color='pink', linewidth=linewidth,
              density=False, label=r'puppi jets pT')
    plt.hist(pt_jets_truth_total, bins=40, range=(0, 400), histtype='step', color='black', linewidth=linewidth,
              density=False, label=r'truth jets pT')
    plt.hist(pt_jets_CHS_total, bins=40, range=(0, 400), histtype='step', color='red', linewidth=linewidth,
              density=False, label=r'CHS jets pT')
    
    plt.xlabel(r"GeV")
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    plt.savefig("JetPT.pdf")

    fig = plt.figure(figsize=(10, 8))
    pt_jets_pf_total = np.array(mass_jets_pf)
    pt_jets_pred_total = np.array(mass_jets_pred)
    pt_jets_puppi_total = np.array(mass_jets_puppi)
    pt_jets_truth_total = np.array(mass_jets_truth)
    pt_jets_CHS_total = np.array(mass_jets_CHS)
    plt.hist(pt_jets_pf_total, bins=40, range=(0, 200), histtype='step', color='blue', linewidth=linewidth,
              density=False, label=r'PF jets Mass')
    plt.hist(pt_jets_pred_total, bins=40, range=(0, 200), histtype='step', color='green', linewidth=linewidth,
              density=False, label=r'SSL jets Mass')
    plt.hist(pt_jets_puppi_total, bins=40, range=(0, 200), histtype='step', color='pink', linewidth=linewidth,
              density=False, label=r'puppi jets Mass')
    plt.hist(pt_jets_truth_total, bins=40, range=(0, 200), histtype='step', color='black', linewidth=linewidth,
              density=False, label=r'truth jets Mass')
    plt.hist(pt_jets_CHS_total, bins=40, range=(0, 200), histtype='step', color='red', linewidth=linewidth,
              density=False, label=r'CHS jets Mass')
    
    plt.xlabel(r"GeV")
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    plt.savefig("JetMass.pdf")

    fig = plt.figure(figsize=(10, 8))
    eta_jets_pf_total = np.array(eta_jets_pf)
    eta_jets_pred_total = np.array(eta_jets_pred)
    eta_jets_puppi_total = np.array(eta_jets_puppi)
    eta_jets_truth_total = np.array(eta_jets_truth)
    eta_jets_CHS_total = np.array(eta_jets_CHS)
    plt.hist(eta_jets_pf_total, bins=40, range=(-3, 3), histtype='step', color='blue', linewidth=linewidth,
              density=False, label=r'PF jets eta')
    plt.hist(eta_jets_pred_total, bins=40, range=(-3, 3), histtype='step', color='green', linewidth=linewidth,
              density=False, label=r'SSL jets eta')
    plt.hist(eta_jets_puppi_total, bins=40, range=(-3, 3), histtype='step', color='pink', linewidth=linewidth,
              density=False, label=r'puppi jets eta')
    plt.hist(eta_jets_truth_total, bins=40, range=(-3, 3), histtype='step', color='black', linewidth=linewidth,
              density=False, label=r'truth jets eta')
    plt.hist(eta_jets_CHS_total, bins=40, range=(-3, 3), histtype='step', color='red', linewidth=linewidth,
              density=False, label=r'CHS jets eta')
    
    plt.xlabel(r"eta")
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    plt.savefig("JetEta.pdf")

    fig = plt.figure(figsize=(10, 8))
    phi_jets_pf_total = np.array(phi_jets_pf)
    phi_jets_pred_total = np.array(phi_jets_pred)
    phi_jets_puppi_total = np.array(phi_jets_puppi)
    phi_jets_truth_total = np.array(phi_jets_truth)
    phi_jets_CHS_total = np.array(phi_jets_CHS)
    plt.hist(phi_jets_pf_total, bins=40, range=(-3, 3), histtype='step', color='blue', linewidth=linewidth,
              density=False, label=r'PF jets phi')
    plt.hist(phi_jets_pred_total, bins=40, range=(-3, 3), histtype='step', color='green', linewidth=linewidth,
              density=False, label=r'SSL jets phi')
    plt.hist(phi_jets_puppi_total, bins=40, range=(-3, 3), histtype='step', color='pink', linewidth=linewidth,
              density=False, label=r'puppi jets phi')
    plt.hist(phi_jets_truth_total, bins=40, range=(-3, 3), histtype='step', color='black', linewidth=linewidth,
              density=False, label=r'truth jets phi')
    plt.hist(phi_jets_CHS_total, bins=40, range=(-3, 3), histtype='step', color='red', linewidth=linewidth,
              density=False, label=r'CHS jets phi')
    
    plt.xlabel(r"phi")
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
    plt.savefig("JetPhi.pdf")

    fig = plt.figure(figsize=(10, 8))
    neutral_weight_total = np.array(neu_weight)
    neutral_puweight_total = np.array(neu_puppiweight)
    chlv_weight_total = np.array(chlv_weight)
    chpu_weight_total = np.array(chpu_weight)
    chlv_puweight_total = np.array(chlv_puppiweight)
    chpu_puweight_total = np.array(chpu_puppiweight)
    print(chlv_weight_total[:100])
    neutral_weight_total = np.array(neu_weight)
    plt.hist(neutral_weight_total, bins=40, range=(0, 1), histtype='step', color='blue', linewidth=linewidth,
              density=True, label=r'Neutral particle weight')
    plt.hist(chlv_weight_total, bins=40, range=(0, 1), histtype='step', color='green', linewidth=linewidth,
              density=True, label=r'Charged LV particle weight')
    plt.hist(chpu_weight_total, bins=40, range=(0, 1), histtype='step', color='pink', linewidth=linewidth,
              density=True, label=r'Charged PU particle weight')
    
    
    plt.xlabel(r"SSL weight")
    plt.ylabel('density')
    plt.legend()
    plt.show()
    plt.savefig("GNNweight.pdf")

    fig = plt.figure(figsize=(10, 8))
    plt.hist(neutral_puweight_total, bins=40, range=(0, 1), histtype='step', color='blue', linewidth=linewidth,
              density=True, label=r'Neutral particle weight')
    plt.hist(chlv_puweight_total, bins=40, range=(0, 1), histtype='step', color='green', linewidth=linewidth,
              density=True, label=r'Charged LV particle weight')
    plt.hist(chpu_puweight_total, bins=40, range=(0, 1), histtype='step', color='pink', linewidth=linewidth,
              density=True, label=r'Charged PU particle weight')
    plt.xlabel(r"puppi weight")
    plt.ylabel('density')
    plt.legend()
    plt.show()
    plt.savefig("GNNpuppi.pdf")

    fig = plt.figure(figsize=(10, 8))
    a = plt.hist2d(neutral_puweight_total,neutral_weight_total, bins=20, range=[[0,1],[0,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('Neutral Particles')
    plt.xlabel(r'puppi weight')
    plt.ylabel(r'SSL weight')
    plt.savefig("neutral2d.pdf")

    fig = plt.figure(figsize=(10, 8))
    a = plt.hist2d(chlv_puweight_total,chlv_weight_total, bins=20, range=[[0,1],[0,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('Charged LV Particles')
    plt.xlabel(r'puppi weight')
    plt.ylabel(r'SSL weight')
    plt.savefig("chlv2d.pdf")

    fig = plt.figure(figsize=(10, 8))
    a = plt.hist2d(chpu_puweight_total,chpu_weight_total, bins=20, range=[[0,1],[0,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('Charged PU Particles')
    plt.xlabel(r'puppi weight')
    plt.ylabel(r'SSL weight')
    plt.savefig("chpu2d.pdf")

    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_pred0])
    mass_truth = np.array([getattr(perf, "mass_truth")
                         for perf in performances_jet_pred0])
    a = plt.hist2d(mass_truth,mass_diff, bins=20, range=[[0,30],[-1,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('SSL Jets')
    plt.xlabel(r'mass [GeV]')
    plt.ylabel(r'mass resolution $(m_{reco} - m_{truth})/m_{truth}$')
    plt.savefig("SSLMassReso2d.pdf")

    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_puppi])
    mass_truth = np.array([getattr(perf, "mass_truth")
                         for perf in performances_jet_puppi])
    a = plt.hist2d(mass_truth,mass_diff, bins=20, range=[[0,30],[-1,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('PUPPI Jets')
    plt.xlabel(r'mass [GeV]')
    plt.ylabel(r'mass resolution $(m_{reco} - m_{truth})/m_{truth}$')
    plt.savefig("PUPPIMassReso2d.pdf")

    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_puppi_wcut])
    mass_truth = np.array([getattr(perf, "mass_truth")
                         for perf in performances_jet_puppi_wcut])
    a = plt.hist2d(mass_truth,mass_diff, bins=20, range=[[0,30],[-1,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('PF Jets')
    plt.xlabel(r'mass [GeV]')
    plt.ylabel(r'mass resolution $(m_{reco} - m_{truth})/m_{truth}$')
    plt.savefig("PFMassReso2d.pdf")

    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_CHS])
    mass_truth = np.array([getattr(perf, "mass_truth")
                         for perf in performances_jet_CHS])
    a = plt.hist2d(mass_truth,mass_diff, bins=20, range=[[0,30],[-1,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('CHS Jets')
    plt.xlabel(r'mass [GeV]')
    plt.ylabel(r'mass resolution $(m_{reco} - m_{truth})/m_{truth}$')
    plt.savefig("CHSMassReso2d.pdf")

    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_pred0])
    pt_truth = np.array([getattr(perf, "pt_truth")
                         for perf in performances_jet_pred0])
    a = plt.hist2d(pt_truth,mass_diff, bins=15, range=[[0,1500],[-1,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('SSL Jets')
    plt.xlabel(r'pt [GeV]')
    plt.ylabel(r'mass resolution $(m_{reco} - m_{truth})/m_{truth}$')
    plt.savefig("SSLMassReso2dvspT.pdf")

    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_puppi])
    pt_truth = np.array([getattr(perf, "pt_truth")
                         for perf in performances_jet_puppi])
    a = plt.hist2d(pt_truth,mass_diff, bins=15, range=[[0,1500],[-1,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('PUPPI Jets')
    plt.xlabel(r'pt [GeV]')
    plt.ylabel(r'mass resolution $(m_{reco} - m_{truth})/m_{truth}$')
    plt.savefig("PUPPIMassReso2dvspT.pdf")

    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_puppi_wcut])
    pt_truth = np.array([getattr(perf, "pt_truth")
                         for perf in performances_jet_puppi_wcut])
    a = plt.hist2d(pt_truth,mass_diff, bins=15, range=[[0,1500],[-1,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('PF Jets')
    plt.xlabel(r'pt [GeV]')
    plt.ylabel(r'mass resolution $(m_{reco} - m_{truth})/m_{truth}$')
    plt.savefig("PFMassReso2dvspT.pdf")

    fig = plt.figure(figsize=(10, 8))
    mass_diff = np.array([getattr(perf, "mass_diff")
                         for perf in performances_jet_CHS])
    pt_truth = np.array([getattr(perf, "pt_truth")
                         for perf in performances_jet_CHS])
    a = plt.hist2d(pt_truth,mass_diff, bins=15, range=[[0,1500],[-1,1]], 
              norm=mpl.colors.LogNorm())
    plt.colorbar(a[3])
    plt.title('CHS Jets')
    plt.xlabel(r'pt [GeV]')
    plt.ylabel(r'mass resolution $(m_{reco} - m_{truth})/m_{truth}$')
    plt.savefig("CHSMassReso2dvspT.pdf")




    # %matplotlib inline
    fig = plt.figure(figsize=(10, 8))

    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_pred0])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='blue', linewidth=linewidth, 
             density=True, label=r'Semi-supevised, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pt_diff)))+str(len(pt_diff)))
    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_puppi])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='green', linewidth=linewidth, 
             density=True, label=r'PUPPI, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pt_diff)))+str(len(pt_diff)))
    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_puppi_wcut])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='red', linewidth=linewidth, 
             density=True, label=r'PF, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pt_diff)))+str(len(pt_diff)))
    pt_diff = np.array([getattr(perf, "pt_diff")
                       for perf in performances_jet_CHS])
    plt.hist(pt_diff, bins=40, range=(-0.3, 0.3), histtype='step', color='orange', linewidth=linewidth, 
             density=True, label=r'CHS, $\mu={:10.3f}$, $\sigma={:10.3f}$, counts:'.format(*(getStat(pt_diff)))+str(len(pt_diff)))
    # plt.xlim(0,40)
    plt.ylim(0, 10)
    plt.xlabel(r"Jet $p_{T}$ $(p^{reco}_{T} - p^{truth}_{T})/p^{truth}_{T}$")
    plt.ylabel('density')
    plt.rc('legend', fontsize=fontsize)
    plt.legend()
    plt.show()
    plt.savefig("Jet_pT_diff.pdf")

    # MET resolution
    # %matplotlib inline

    fig = plt.figure(figsize=(10, 8))
    mets_diff = (np.array(mets_pred0) - np.array(mets_truth))
    plt.hist(mets_diff, bins=30, range=(-30, 30), histtype='step', color='blue', linewidth=linewidth,
             density=True, label=r'Semi-supervised, $\mu={:10.2f}$, $\sigma={:10.2f}$'.format(*(getStat(mets_diff))))
    mets_diff = (np.array(mets_puppi) - np.array(mets_truth))
    plt.hist(mets_diff, bins=30, range=(-30, 30), histtype='step', color='green', linewidth=linewidth,
             density=True, label=r'PUPPI, $\mu={:10.2f}$, $\sigma={:10.2f}$'.format(*(getStat(mets_diff))))
    mets_diff = (np.array(mets_puppi_wcut) - np.array(mets_truth))
    plt.hist(mets_diff, bins=30, range=(-30, 30), histtype='step', color='red', linewidth=linewidth,
             density=True, label=r'PF, $\mu={:10.2f}$, $\sigma={:10.2f}$'.format(*(getStat(mets_diff))))

    plt.xlabel(r"$p^{miss, reco}_{T} - p^{miss, truth}_{T}$ [GeV]")
    plt.ylabel('density')
    plt.ylim(0, 0.04)
    plt.legend()
    plt.show()
    plt.savefig("MET_diff.pdf")

    # more plots to be included


if __name__ == '__main__':
    modelname = "test/best_valid_model_nPU20_deeper.pt"
    filelists = ["../data_pickle/dataset_graph_puppi_test_Wjets4000"]
    main(modelname, filelists)
