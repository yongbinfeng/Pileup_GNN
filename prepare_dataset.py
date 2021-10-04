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
from timeit import default_timer as timer



# select pfcands from original root and convert to a dataframe'
def gen_dataframe(num_event):
    tree = uproot.open("myNanoProdMc2016_NANO_1_puppi.root")["Events"]
    pfcands = tree.arrays(tree.keys('PF_*'), entry_start=0, entry_stop=num_event)

    df_list = []
    for i in range(num_event):
        event = pfcands[i]
        # eliminate those with eta more than 2.5
        eta_eliminate = abs(event['PF_eta']) < 2.5
        event = event[eta_eliminate]
        selected_features = ['PF_eta', 'PF_phi', 'PF_pt', 'PF_pdgId', 'PF_charge', 'PF_puppiWeight', 'PF_weight']
        event_nPF = ak.to_numpy(event['PF_eta']).size
        # selected_particles = random.sample(range(event_nPF))
        pf_chosen = event[selected_features]

        df_pfcands = ak.to_pandas(pf_chosen)
        df_pfcands['PF_pt'] = np.log(df_pfcands['PF_pt'])
        df_list.append(df_pfcands)

    return df_list


# reconstruct = 0 means initialize; 1 means reconstruct the existing graph, -1 means resample with the same graph
def prepare_dataset(num_event):
    data_list = []

    df_list = gen_dataframe(num_event)

    for num in range(len(df_list)):
        df_pfcands = df_list[num]
        LV_index = df_pfcands[(df_pfcands['PF_puppiWeight'] == 1) & (df_pfcands['PF_charge'] != 0) & (
                    df_pfcands['PF_pt'] > 0.8)].index.codes[0]
        PU_index = df_pfcands[(df_pfcands['PF_puppiWeight'] == 0) & (df_pfcands['PF_charge'] != 0) & (
                    df_pfcands['PF_pt'] > 0.8)].index.codes[0]
        if LV_index.shape[0] < 10 or PU_index.shape[0] < 10:
            continue
        Neutral_index = df_pfcands[(df_pfcands['PF_charge'] == 0)].index.codes[0]
        Charge_index = df_pfcands[(df_pfcands['PF_charge'] != 0)].index.codes[0]

        # label of samples
        label = df_pfcands.loc[:, ['PF_puppiWeight']].to_numpy()
        label = torch.from_numpy(label).view(-1)
        label = label.type(torch.long)

        # PF_weight for puppi
        weight = torch.from_numpy(df_pfcands.loc[:, ['PF_weight']].to_numpy())

        # node features
        node_features = df_pfcands.drop(df_pfcands.loc[:, ['PF_charge']], axis=1).to_numpy()
        node_features = torch.from_numpy(node_features)
        node_features = node_features.type(torch.float32)

        # set the neutral puppiWeight to default
        node_features[[Neutral_index.tolist()], [4]] = 2
        # set the charge pdgId for one hot encoding later
        node_features[[Charge_index.tolist()], [3]] = 0

        # get index for mask training and testing samples for pdgId(3) and puppiWeight(4)
        photon_index = df_pfcands[(df_pfcands['PF_pdgId'] == 22)].index.codes[0]
        # hadron_index = df_pfcands[(df_pfcands['PF_pdgId'] == 130)].index.codes[0]
        photon_portion = (photon_index.shape[0] + 1) / (Neutral_index.shape[0] + 1)
        # hadron_portion = 1 - photon_portion

        # one hot encoding for pdgId and puppiWeight
        pdgId = node_features[:, 3]
        puppiWeight = node_features[:, 4]

        photon_indices = pdgId == 22
        pdgId[photon_indices] = 1
        hadron_indices = pdgId == 130
        pdgId[hadron_indices] = 2

        pdgId = pdgId.type(torch.long)
        puppiWeight = puppiWeight.type(torch.long)

        pdgId_one_hot = torch.nn.functional.one_hot(pdgId)
        puppiWeight_one_hot = torch.nn.functional.one_hot(puppiWeight)

        pdgId_one_hot = pdgId_one_hot.type(torch.float32)
        puppiWeight_one_hot = puppiWeight_one_hot.type(torch.float32)
        node_features = torch.cat((node_features[:, 0:3], pdgId_one_hot, puppiWeight_one_hot, weight), 1)

        # node_features = node_features.type(torch.float32)
        # construct edge index for graph

        phi = df_pfcands['PF_phi'].to_numpy().reshape((-1, 1))
        eta = df_pfcands['PF_eta'].to_numpy().reshape((-1, 1))

        dist_phi = distance.cdist(phi, phi, 'cityblock')
        indices = np.where(dist_phi > pi)
        temp = np.ceil((dist_phi[indices] - pi) / (2 * pi)) * (2 * pi)
        dist_phi[indices] = dist_phi[indices] - temp
        dist_eta = distance.cdist(eta, eta, 'cityblock')

        dist = np.sqrt(dist_phi ** 2 + dist_eta ** 2)
        edge_source = np.where((dist < 0.8) & (dist != 0))[0]
        edge_target = np.where((dist < 0.8) & (dist != 0))[1]

        """
        # fourier feature mapping
        B_eta = torch.randint(0, 10, (1, 10), dtype=torch.float)
        B_phi = torch.randint(0, 100, (1, 1), dtype=torch.float)
        alpha_phi = 1
        alpha_eta = 1
        pre_ffm_phi = node_features[:, 1].clone().view(-1, 1)
        pre_ffm_eta = node_features[:, 0].clone().view(-1, 1)
        pre_ffm_phi = pre_ffm_phi.type(torch.float)
        pre_ffm_eta = pre_ffm_eta.type(torch.float)

        ffm_phi_temp = (2 * np.pi * alpha_phi * pre_ffm_phi) @ B_phi
        ffm_eta_temp = (2 * np.pi * alpha_eta * pre_ffm_eta) @ B_eta
        ffm_phi = torch.cat((torch.sin(ffm_phi_temp), torch.cos(ffm_phi_temp)), dim=1)
        ffm_eta = torch.cat((torch.sin(ffm_eta_temp), torch.cos(ffm_eta_temp)), dim=1)
        ffm_phi_eta = torch.cat((ffm_phi, ffm_eta), dim=1)
        node_features = torch.cat((ffm_phi_eta, node_features[:, 2:]), dim=1)
        """

        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)

        graph = Data(x=node_features, edge_index=edge_index, y=label)
        graph.LV_index = LV_index
        graph.PU_index = PU_index
        graph.Neutral_index = Neutral_index
        graph.Charge_index = Charge_index
        graph.photon_portion = photon_portion
        graph.num_classes = 2
        data_list.append(graph)

    return data_list


def main():
    start = timer()
    num_events = 2000
    dataset = prepare_dataset(num_events)

    with open("dataset_graph_puppi_" + str(num_events), "wb") as fp:
        pickle.dump(dataset, fp)

    end = timer()
    program_time = end - start
    print("generating graph time " + str(program_time))


if __name__ == '__main__':
    main()







