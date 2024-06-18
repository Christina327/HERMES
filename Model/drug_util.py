import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GraphDataset(InMemoryDataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for atom_features, sparse_adj in graphs_dict:
            GCNData = DATA.Data(
                x=torch.Tensor(atom_features),
                edge_index=torch.LongTensor(sparse_adj))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA.to(DEVICE)


def calculate_graph_feat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))

    return [feat_mat, adj_index]


def drug_feature_extract(drug_data):
    drug_data = pd.DataFrame(drug_data).T
    drug_feat = [[] for _ in range(len(drug_data))]
    for i in range(len(drug_feat)):
        feat_mat, adj_list = drug_data.iloc[i]
        drug_feat[i] = calculate_graph_feat(feat_mat, adj_list)
    return drug_feat
