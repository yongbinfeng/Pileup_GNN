import matplotlib.pyplot as plt
import numpy as np
import math
from math import pi
import torch
import random
from torch_geometric.data import Data
import pickle
from scipy.spatial import distance
import h5py
from scipy import stats
import copy

np.random.seed(0)
from timeit import default_timer as timer


def cal_Median_LeftRMS(x):
    """
    Given on 1d np array x, return the median and the left RMS
    """
    median = np.median(x)
    x_diff = x - median
    # only look at differences on the left side of median
    x_diffLeft = x_diff[x_diff < 0]
    rmsLeft = np.sqrt(np.sum(x_diffLeft ** 2) / x_diffLeft.shape[0])
    return median, rmsLeft


def buildConnections(eta, phi):
    """
    build the Graph based on the deltaEta and deltaPhi of input particles
    """
    dist_phi = distance.cdist(phi, phi, 'cityblock')
    indices = np.where(dist_phi > pi)
    temp = np.ceil((dist_phi[indices] - pi) / (2 * pi)) * (2 * pi)
    dist_phi[indices] = dist_phi[indices] - temp
    dist_eta = distance.cdist(eta, eta, 'cityblock')
    dist = np.sqrt(dist_phi ** 2 + dist_eta ** 2)
    edge_source = np.where((dist < 0.8) & (dist != 0))[0]
    edge_target = np.where((dist < 0.8) & (dist != 0))[1]
    return edge_source, edge_target


def prepare_dataset(num_event):
    num_file = max(int(num_event / 100), 1)
    particles = []
    for i in range(num_file):
        if i == num_file - 1:
            cur_num_event = num_event - i * 100
        else:
            cur_num_event = 100

        filepath = "ZnunuPlusJet_13TeV_80PU_withUnderlyingEvent/ZnunuPlusJet_13TeV_80PU_withUnderlyingEvent_" \
                   + str(i) + ".h5"
        event = h5py.File(filepath, "r")
        temp_particles = list(np.array(event['Particles'][0:cur_num_event]))

        particles = particles + temp_particles


    data_list = []
    for i in range(num_event):
        if i % 1 == 0:
            print("processed {} events".format(i))
        event = particles[i]
        # remove zero-padded particles by cutting on pt>0
        # also remove the neutrinos from the input particle list
        event = event[(event[:, 4] > 0) & (event[:, 18] == 0)]

        print("done")
        # calculate PUPPI weights based on alphaCh_1
        isChgPU = ((event[:, 7] != 0) & (event[:, 17] != 0))
        alphas_ChgPU = event[isChgPU, 11]
        # around 60% of charged PU particles have no neighbor within the cone
        # remove these particles from the median and RMS calculation
        alphas_ChgPU = alphas_ChgPU[alphas_ChgPU > -1e5]
        alphaMedian, alphaRMS = cal_Median_LeftRMS(alphas_ChgPU)

        alphasAll = event[:, 11]
        chi2 = np.heaviside(alphasAll - alphaMedian, 0) * (alphasAll - alphaMedian) ** 2 / alphaRMS ** 2
        # puppi weights calculated from chi2
        weights = stats.chi2.cdf(chi2, 1)
        plt.hist(weights, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2])
        plt.show()
        plt.clf()
        print("done!")

        ## to mimic the standard setup, set the charged puppi weights either 0 or 1, depending on the true flag
        # puppiWeights = copy.deepcopy(weights)
        # puppiWeights[(event[:,7]!=0) &  (event[:, 17] == 0)] = 1
        # puppiWeights[(event[:,7]!=0) &  (event[:, 17] != 0)] = 0

        # fromLV information, to set another col independent from puppiWeights
        fromLV = np.zeros(event.shape[0])
        fromLV[(event[:, 7] != 0) & (event[:, 17] == 0)] = 1
        fromLV[(event[:, 7] != 0) & (event[:, 17] != 0)] = 0
        # for neutrals, set fromPV to 2
        fromLV[event[:, 7] == 0] = 2

        LV_index = np.where((event[:, 7] != 0) & (fromLV != 0) & (event[:, 4] > 0.5))[0]
        PU_index = np.where((event[:, 7] != 0) & (fromLV == 0) & (event[:, 4] > 0.5))[0]
        Neutral_index = np.where((event[:, 7] == 0))[0]
        Charge_index = np.where((event[:, 7] != 0))[0]

        # calculate deltaR
        eta = event[:, 5:6]
        phi = event[:, 6:7]
        edge_source, edge_target = buildConnections(eta, phi)
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        print("done!!")

        # node features
        eta = event[:, 5]
        phi = event[:, 6]
        pt = event[:, 4]
        chg = event[:, 7]
        # no charge information as full simulation
        node_features = np.stack((eta, phi, pt, fromLV, weights), axis=1)
        node_features = torch.from_numpy(node_features)
        node_features = node_features.type(torch.float32)

        # one hot for fromLV
        fromLV = node_features[:, 3].type(torch.long)
        fromLV_onehot = torch.nn.functional.one_hot(fromLV)
        fromLV_onehot = fromLV_onehot.type(torch.float32)
        node_features = torch.cat((node_features[:, 0:3], fromLV_onehot, node_features[:, -1].view(-1, 1)), 1)

        # truth label
        label = torch.from_numpy((event[:, 17] == 0))
        label = label.type(torch.long)

        graph = Data(x=node_features, edge_index=edge_index, y=label)
        graph.LV_index = LV_index
        graph.PU_index = PU_index
        graph.Neutral_index = Neutral_index
        graph.Charge_index = Charge_index
        graph.num_classes = 2
        print("done!!!")
        data_list.append(graph)

    return data_list


def main():
    start = timer()
    num_events = 200
    dataset = prepare_dataset(num_events)

    with open("dataset_ggnn_onehot_" + str(num_events), "wb") as fp:
        pickle.dump(dataset, fp)

    end = timer()
    program_time = end - start
    print("generating graph time " + str(program_time))


if __name__ == '__main__':
    main()
