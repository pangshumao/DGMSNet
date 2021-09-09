import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from torch.autograd import Variable

spine_graph = {
    0: [0], # background
    1: [1, 6], # L1
    2: [2, 6, 7], # L2
    3: [3, 7, 8], # L3
    4: [4, 8, 9], # L4
    5: [5, 9, 10], # L5
    6: [6, 1, 2], # L1/L2
    7: [7, 2, 3], # L2/L3
    8: [8, 3, 4], # L3/L4
    9: [9, 4, 5], # L4/L5
    10: [10, 5]} # L5/S1


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj)) # return a adjacency matrix of adj ( type is numpy)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) #
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.todense()

def row_norm(inputs):
    outputs = []
    for x in inputs:
        xsum = x.sum()
        x = x / xsum
        outputs.append(x)
    return outputs


def normalize_adj_torch(adj):
    # print(adj.size())
    if len(adj.size()) == 4:
        new_r = torch.zeros(adj.size()).type_as(adj)
        for i in range(adj.size(1)):
            adj_item = adj[0,i]
            rowsum = adj_item.sum(1)
            d_inv_sqrt = rowsum.pow_(-0.5)
            d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            r = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj_item), d_mat_inv_sqrt)
            new_r[0,i,...] = r
        return new_r
    rowsum = adj.sum(1)
    d_inv_sqrt = rowsum.pow_(-0.5)
    d_inv_sqrt[torch.isnan(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    r = torch.matmul(torch.matmul(d_mat_inv_sqrt,adj),d_mat_inv_sqrt)
    return r



if __name__ == '__main__':
    # a= row_norm(cihp2pascal_adj)
    # print(a)
    # print(cihp2pascal_adj)

    a = row_norm(preprocess_adj(spine_graph))
    print(a)
    print(preprocess_adj(spine_graph))

    # cihp_adj = preprocess_adj(cihp_graph)
    # adj1_ = torch.from_numpy(cihp_adj).float()
    # adj1 = adj1_.unsqueeze(0).unsqueeze(0).cuda()
    # adj1_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20)
