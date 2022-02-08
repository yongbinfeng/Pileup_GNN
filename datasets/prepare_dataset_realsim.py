from timeit import default_timer as timer
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import math
from math import pi
import torch
import random
from torch_geometric.data import Data
import pickle
from scipy.spatial import distance

np.random.seed(0)


# select pfcands from original root and convert to a dataframe'
def gen_dataframe(num_event, num_start=0):
    print(f"reading events from {num_start} to {num_start+num_event}")
    tree = uproot.open(
        "/depot/cms/private/users/feng356/SSLPUPPI_fullSim/output_1.root")["Events"]
    pfcands = tree.arrays(
        tree.keys('PF_*'), entry_start=num_start, entry_stop=num_event + num_start)

    df_list = []
    #
    # todo: this loop can probably be removed
    #
    for i in range(num_event):
        event = pfcands[i]
        # eliminate those with eta more than 2.5
        eta_eliminate = abs(event['PF_eta']) < 2.5
        event = event[eta_eliminate]
        #
        # todo: add more features here
        #
        selected_features = ['PF_eta', 'PF_phi', 'PF_pt',
                             'PF_pdgId', 'PF_charge', 'PF_puppiWeight']
        event_nPF = ak.to_numpy(event['PF_eta']).size
        pf_chosen = event[selected_features]

        df_pfcands = ak.to_pandas(pf_chosen)
        #df_pfcands['PF_pt'] = np.log(df_pfcands['PF_pt'])
        df_list.append(df_pfcands)

    return df_list


def prepare_dataset(num_event, num_start=0):
    data_list = []

    df_list = gen_dataframe(num_event, num_start)

    PTCUT = 0.5

    for num in range(len(df_list)):
        if num % 100 == 0:
            print(f"processed {num} events")
        #
        # toDO: this for loop can probably be also removed
        # and the gen_dataframe can be merged inside this function
        #
        df_pfcands = df_list[num]
        LV_index = df_pfcands[(df_pfcands['PF_puppiWeight'] > 0.99) & (df_pfcands['PF_charge'] != 0) & (
            df_pfcands['PF_pt'] > PTCUT)].index.codes[0]
        PU_index = df_pfcands[(df_pfcands['PF_puppiWeight'] < 0.01) & (df_pfcands['PF_charge'] != 0) & (
            df_pfcands['PF_pt'] > PTCUT)].index.codes[0]
        #print("LV index", LV_index)
        #print("PU index", PU_index)
        if LV_index.shape[0] < 5 or PU_index.shape[0] < 50:
            continue
        Neutral_index = df_pfcands[(
            df_pfcands['PF_charge'] == 0)].index.codes[0]
        Charge_index = df_pfcands[(
            df_pfcands['PF_charge'] != 0)].index.codes[0]

        # label of samples
        label = df_pfcands.loc[:, ['PF_puppiWeight']].to_numpy()
        label = torch.from_numpy(label).view(-1)
        label = label.type(torch.long)

        # node features
        node_features = df_pfcands.drop(
            df_pfcands.loc[:, ['PF_charge']], axis=1).to_numpy()
        node_features = torch.from_numpy(node_features)
        node_features = node_features.type(torch.float32)

        # set the charge pdgId for one hot encoding later
        node_features[[Charge_index.tolist()], 3] = 0
        # get index for mask training and testing samples for pdgId(3) and puppiWeight(4)
        # one hot encoding for pdgId and puppiWeight
        pdgId = node_features[:, 3]
        photon_indices = (pdgId == 22)
        pdgId[photon_indices] = 1
        hadron_indices = (pdgId == 130)
        pdgId[hadron_indices] = 2
        pdgId = pdgId.type(torch.long)
        pdgId_one_hot = torch.nn.functional.one_hot(pdgId)
        pdgId_one_hot = pdgId_one_hot.type(torch.float32)

        # set the neutral puppiWeight to default
        node_features[[Neutral_index.tolist()], 4] = 2
        puppiWeight = node_features[:, 4]
        puppiWeight = puppiWeight.type(torch.long)
        puppiWeight_one_hot = torch.nn.functional.one_hot(puppiWeight)
        puppiWeight_one_hot = puppiWeight_one_hot.type(torch.float32)

        node_features = torch.cat(
            (node_features[:, 0:3], pdgId_one_hot, puppiWeight_one_hot), 1)

        if num == 0:
            print("pdgId dimensions: ", pdgId_one_hot.shape)
            print("puppi weights dimensions: ", puppiWeight_one_hot.shape)
            print("node_features dimension: ", node_features.shape)

        # node_features = node_features.type(torch.float32)
        # construct edge index for graph

        phi = df_pfcands['PF_phi'].to_numpy().reshape((-1, 1))
        eta = df_pfcands['PF_eta'].to_numpy().reshape((-1, 1))

        dist_phi = distance.cdist(phi, phi, 'cityblock')
        # deal with periodic feature of phi
        indices = np.where(dist_phi > pi)
        temp = np.ceil((dist_phi[indices] - pi) / (2 * pi)) * (2 * pi)
        dist_phi[indices] = dist_phi[indices] - temp
        dist_eta = distance.cdist(eta, eta, 'cityblock')

        dist = np.sqrt(dist_phi ** 2 + dist_eta ** 2)
        edge_source = np.where((dist < 0.4) & (dist != 0))[0]
        edge_target = np.where((dist < 0.4) & (dist != 0))[1]

        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)

        graph = Data(x=node_features, edge_index=edge_index, y=label)
        graph.LV_index = LV_index
        graph.PU_index = PU_index
        graph.Neutral_index = Neutral_index
        graph.Charge_index = Charge_index
        graph.num_classes = 2
        data_list.append(graph)

    return data_list


def main():
    start = timer()
    num_events_train = 100
    dataset_train = prepare_dataset(num_events_train)
    with open("dataset_graph_puppi_" + str(num_events_train), "wb") as fp:
        pickle.dump(dataset_train, fp)

    num_events_valid = 50
    dataset_valid = prepare_dataset(num_events_valid, num_events_train)
    with open("dataset_graph_puppi_" + str(num_events_valid), "wb") as fp:
        pickle.dump(dataset_valid, fp)


    end = timer()
    program_time = end - start
    print("generating graph time " + str(program_time))


if __name__ == '__main__':
    main()
