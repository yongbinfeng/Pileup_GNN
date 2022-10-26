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
#        "/depot/cms/private/users/gpaspala/output_1.root")["Events"]
         "/depot/cms/private/users/gpaspala/ZJetsToNuNu_HT-200To400/output_1.root")["Events"]
    pfcands = tree.arrays(
        tree.keys('PF_*'), entry_start=num_start, entry_stop=num_event + num_start)

    genparts= tree.arrays(
        tree.keys('packedGenPart_*'), entry_start=num_start, entry_stop=num_event + num_start)

    print (tree.num_entries)

    df_list = []
    df_gen_list = []
    df_pf_raw_list = []

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
        #selected_features = ['PF_eta', 'PF_phi', 'PF_pt',
         #                    'PF_pdgId', 'PF_charge', 'PF_puppiWeight','PF_dz']
        selected_features = ['PF_eta', 'PF_phi', 'PF_pt',
                             'PF_pdgId', 'PF_charge', 'PF_puppiWeight','PF_puppiWeightChg','PF_dz']
        event_nPF = ak.to_numpy(event['PF_eta']).size
        pf_chosen = event[selected_features]

        df_pfcands = ak.to_pandas(pf_chosen)

        #df_pfcands['PF_pt'] = np.log(df_pfcands['PF_pt'])
        df_list.append(df_pfcands)
        df_pf_raw_list.append(df_pfcands)


    for i in range(num_event):
        event = genparts[i]
        # eliminate those with eta more than 2.5 and also neutrinos
        selection = ((abs(event['packedGenPart_eta']) < 2.5) & (abs(event['packedGenPart_pdgId']) != 12) &  (abs(event['packedGenPart_pdgId']) != 14) & (abs(event['packedGenPart_pdgId']) != 16))
        event = event[selection]
        selected_features = ['packedGenPart_eta', 'packedGenPart_phi', 'packedGenPart_pt', 'packedGenPart_pdgId', 'packedGenPart_charge']
        gen_chosen = event[selected_features]
        df_genparts = ak.to_pandas(gen_chosen)
        df_gen_list.append(df_genparts)


    return df_list, df_gen_list,df_pf_raw_list


def prepare_dataset(num_event, num_start=0):
    data_list = []

    df_list, df_gen_list ,df_pf_raw_list = gen_dataframe(num_event, num_start)

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
        #print ("pdgID_one_hot", pdgId_one_hot)
        # set the neutral puppiWeight to default
        node_features[[Neutral_index.tolist()], 4] = 2
        puppiWeight = node_features[:, 4]
        puppiWeight = puppiWeight.type(torch.long)
        puppiWeight_one_hot = torch.nn.functional.one_hot(puppiWeight)
        puppiWeight_one_hot = puppiWeight_one_hot.type(torch.float32)
        columnsNamesArr = df_pfcands.columns.values
        #print ("puppiWeight_one_hot", puppiWeight_one_hot)
        #if num ==0:
           #print("node_features[:,-1:]",node_features[:,-1:])
           #print("node_features:", node_features[:,5:6])
        node_features = torch.cat(
            # (node_features[:, 0:3], pdgId_one_hot, puppiWeight_one_hot), 1)
            (node_features[:, 0:3], pdgId_one_hot,node_features[:,-1:], puppiWeight_one_hot), 1)
            #i(node_features[:, 0:3], pdgId_one_hot,node_features[:,5:6], puppiWeight_one_hot), 1)
            # (node_features[:, 0:4], pdgId_one_hot, puppiWeight_one_hot), 1)

        if num == 0:
            print("pdgId dimensions: ", pdgId_one_hot.shape)
            print("puppi weights dimensions: ", puppiWeight_one_hot.shape)
            print("last dimention: ", node_features[:,-1:].shape)
            print("node_features dimension: ", node_features.shape)
            print("node_features[:, 0:3] dimention: ", node_features[:, 0:3].shape)
            print("node_features dimension: ", node_features.shape)
            print("node_features[:, 6:7]", node_features[:, 6:7].shape) #dz values
            #print("columnsNamesArr", columnsNamesArr)
            #print ("pdgId_one_hot " , pdgId_one_hot)
            #print("node_features[:,-1:]",node_features[:,-1:])
            #print("puppi weights", puppiWeight_one_hot)
            #print("node features: ", node_features)

        # node_features = node_features.type(torch.float32)
        # construct edge index for graph

        phi        = df_pfcands['PF_phi'].to_numpy().reshape((-1, 1))
        eta        = df_pfcands['PF_eta'].to_numpy().reshape((-1, 1))
        #vertexChi2 = df_pfcands['PF_vertexChi2'].to_numpy().reshape((-1, 1))
        #mass       = df_pfcands['PF_mass'].to_numpy().reshape((-1, 1))
        dz         = df_pfcands['PF_dz'].to_numpy().reshape((-1, 1))
        #d0         = df_pfcands['PF_d0'].to_numpy().reshape((-1, 1))
        #vertexNdof = df_pfcands['PF_vertexNdof'].to_numpy().reshape((-1, 1))
        #fromPV     = df_pfcands['PF_fromPV'].to_numpy().reshape((-1, 1))
        #trkChi2 = df_pfcands['PF_trkChi2'].to_numpy().reshape((-1, 1))
        #vertexNormalizedChi2 = df_pfcands['PF_vertexNormalizedChi2'].to_numpy().reshape((-1, 1))
        charge= df_pfcands['PF_charge'].to_numpy().reshape((-1, 1))

        #PF_hasTrackDetails = df_pfcand['PF_hasTrackDetails'].to_numpy().reshape((-1, 1))
       # print ("phi", phi)
        #print ("PF_fromPV", fromPV)
        #print ("PF_trkChi2",trkChi2)
        #print ("PF_vertexNormalizedChi2",vertexNormalizedChi2)
        #print ("vertexNdof",vertexNdof)
       # print ("PF_d0", d0)
        #print ("PF_dz", dz)
       # print ("PF_charge", charge)
       # print ("PF_vertexChi2",vertexChi2)

        df_gencands =df_gen_list[num]
        gen_features = df_gencands.to_numpy()
        gen_features = torch.from_numpy(gen_features)
        gen_features = gen_features.type(torch.float32)

        df_pf_raw = df_pf_raw_list[num]
        pf_rawfeatures = df_pf_raw.to_numpy()
        pf_rawfeatures = torch.from_numpy(pf_rawfeatures)
        pf_rawfeatures= pf_rawfeatures.type(torch.float32)
            #gen_features = torch.cat((gen_features[:,0:4]),1)

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
        graph.GenPart_nump = gen_features
        graph.PFPartraw_nump = pf_rawfeatures
        graph.GenParts = df_gen_list[num]
        data_list.append(graph)

    return data_list


def main():
    start = timer()
    num_events_train = 2000
    dataset_train = prepare_dataset(num_events_train)
    with open("dataset_graph_puppi_" + str(num_events_train), "wb") as fp:
        pickle.dump(dataset_train, fp)

    num_events_test=2000
    dataset_test = prepare_dataset(num_events_test)
    with open("dataset_graph_puppi_test_" + str(num_events_test), "wb") as fp:
        pickle.dump(dataset_train, fp)

    num_events_valid = 2000
    dataset_valid = prepare_dataset(num_events_valid, num_events_train)
    with open("dataset_graph_puppi_val_" + str(num_events_valid), "wb") as fp:
        pickle.dump(dataset_valid, fp)

    num_events_valid = 1500
    dataset_valid = prepare_dataset(num_events_valid, num_events_train)
    with open("dataset_graph_puppi_PF_puppiWeightChg_" + str(num_events_valid), "wb") as fp:
        pickle.dump(dataset_valid, fp)

    end = timer()
    program_time = end - start
    print("generating graph time " + str(program_time))

if __name__ == '__main__':
    main()
