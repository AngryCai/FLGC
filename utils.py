import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures, GDC


def encode_onehot(labels):
    n_clz = torch.unique(labels).size(0)
    source = torch.ones((labels.shape[0], 1), dtype=torch.float32)
    labels_onehot = torch.zeros((labels.shape[0], n_clz), dtype=torch.float32)
    labels_onehot.scatter_(dim=1, index=labels.unsqueeze(1), src=source)
    return labels_onehot


def load_data(path="../data/cora/", dataset_name="cora", use_dgc=False, split='public'):
    """
    Load dataset from PyG datasets
    :param path:
    :param dataset_name:
    :param use_dgc: whether DGC is being used
    :return:
    """
    # path='/tmp/Cora'
    # dataset_name='Cora'
    print('Loading {} dataset...'.format(dataset_name))
    dataset = Planetoid(root=path, name=dataset_name, transform=[NormalizeFeatures()], split=split)
    data = dataset.data
    if use_dgc:
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.05),
                  sparsification_kwargs=dict(method='topk', k=128,
                                             dim=0), exact=True)
        data = gdc(data)
    print(data)
    adj = sp.coo_matrix((np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
                        shape=(data.num_nodes, data.num_nodes),
                        dtype=np.float32)
    y_one_hot = encode_onehot(data.y)
    data.y_one_hot = y_one_hot
    data.adj = adj
    return data


def cal_test_acc(y_test, y_pred):
    acc = torch.eq(torch.argmax(y_test, dim=1), torch.argmax(y_pred, dim=1))
    acc = torch.sum(acc) * 1. / y_test.shape[0]
    return acc

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).transpose().tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN_Model model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized
