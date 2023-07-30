"""
test unsupervised FLGC on uci datasets
========================
"""
# # ----------------------
#  using simple data
# # ----------------------
import time

import torch
from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize, scale
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected

from FLGC import UnFLGC
from utils import encode_onehot

# # --------------------------------
#  load sklearn datasets
# # --------------------------------
# X, y = load_wine(return_X_y=True)
X, y = load_iris(return_X_y=True)

# # ==============================
X_ = normalize(X)
sparse_g = kneighbors_graph(X_, n_neighbors=10, include_self=False, mode='connectivity')
g = sparse_g.todense()
coo_g = sparse.coo_matrix(g)
edge_index, edge_fea = from_scipy_sparse_matrix(coo_g)
Y = encode_onehot(torch.from_numpy(y).type(torch.LongTensor))
data = Data(x=torch.from_numpy(scale(X)).type(torch.FloatTensor), edge_index=edge_index,
            y=torch.from_numpy(y).to(torch.long), y_one_hot=Y)
data.edge_index = to_undirected(data.edge_index)
# data = T.NormalizeFeatures()(data)
print(data)
# print(data.contains_self_loops())
# print(data.is_directed())
# model = CS_GCN(regularization_coef=1e-1, K_hop=5, kernel='sigmoid', gamma=0.001).to('cuda:0')

n_clz = data.y_one_hot.shape[1]
# # ==================================
# # parameter-settings
# iris: nei=10, lamda=100, K=20, alpha=0.001 ==>ACC=97.33
# # ==================================

start_time = time.time()
model = UnFLGC(n_clz, agg='appnp', regularization_coef=100, K_hop=20, alpha=0.001, save_affinity=False)
model.train()
y_pred = model(data)
run_time = time.time() - start_time
y_best, acc, nmi, kappa, ari, fscore, ca = model.cluster_accuracy(data.y.numpy(), y_pred, return_aligned=True)
print('ACC: %.4f, NMI: %.4f, ARI: %.4f' % (acc, nmi, ari))
print('TIME: {:.4f}s'.format(run_time))

