import numpy as np
import torch
import torch.nn.functional as F
import torch_sparse as ts
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score
from sklearn.preprocessing import normalize
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul


class MySGConv(MessagePassing):
    r"""
    Compute the normalized adjacent matrix and its K-hop $A^K$
    Copied form PyG nn.SGConv and ignored linear transform (PyG 1.6.3)
    """

    def __init__(self, K=1, cached=False, add_self_loops=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MySGConv, self).__init__(**kwargs)
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None
        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.lin.reset_parameters()
    #     self._cached_x = None

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels,
                                         self.K)


class PageRankAgg(MessagePassing):

    def __init__(self, K: int = 1, alpha: float = 0.2, dropout: float = 0.,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(PageRankAgg, self).__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x
        for k in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            x = x * (1 - self.alpha)
            x += self.alpha * h

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)


class SemiFLGC(torch.nn.Module):
    """
    semi-supervised FLGC models with closed-form solution
    """

    def __init__(self, agg='appnp', regularization_coef=1e-5, K_hop=1, rvfl_used=False, **kwargs):
        super(SemiFLGC, self).__init__()
        self.regularization_coef = regularization_coef
        self.K_hop = K_hop
        self.agg = agg
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.agg == 'sgc':
            self.sgconv = MySGConv(K=self.K_hop, cached=False).to(device)
        elif self.agg == 'appnp':
            if 'alpha' in kwargs:
                alpha = kwargs['alpha']
            else:
                alpha = 0.1
            self.sgconv = PageRankAgg(K=self.K_hop, alpha=alpha).to(device)
        else:
            raise Exception('no such aggregation!')
        self.rvfl_used = rvfl_used
        self.kwargs = kwargs

    def forward(self, data):
        """
        :param data: feed into a PyG Data that additionally holds a one-hot label matrix and a sparse adjacent matrix
        :return:
        """
        x_gconv = self.sgconv(data.x, data.edge_index, data.edge_attr)
        self.embedding = x_gconv
        closed_form_solution = self.cal_closed_form_solution(x_gconv, data.y_one_hot, data.train_mask)
        y_pred = torch.matmul(x_gconv, closed_form_solution)
        self.closed_form_solution = closed_form_solution
        return y_pred

    def cal_closed_form_solution(self, x_gconv, y_one_hot, train_mask):
        """
        :param x_gconv: dense tensor
        :param y_one_hot: dense label matrix
        :param train_mask: 1D dense tensor
        :return:
        """
        # mask = torch.diag(data.train_mask.float())
        # mask = ts.SparseTensor.eye(train_mask.size(0)).float()
        row = torch.arange(0, train_mask.shape[0], dtype=torch.long)
        col = torch.arange(0, train_mask.shape[0], dtype=torch.long)
        # val = train_mask.float().dense()
        mask = ts.SparseTensor(row=row, col=col, value=train_mask.float()).coalesce()
        Y_train = y_one_hot * torch.unsqueeze(train_mask, 1)
        # # support only (Sparse_A X Dense_B) or (Sparse_A X Sparse_B).
        # # Thus, for the case of (Dense_A X Sparse_B), considering (Sparse_B.T X Dense_A.T).T
        temp_a = ts.matmul(mask, x_gconv).transpose(1, 0)  # # X.T*M
        before_inv = torch.matmul(temp_a, x_gconv) + self.regularization_coef * torch.eye(x_gconv.shape[1]).float()
        temp_left = torch.inverse(before_inv)
        temp_right = torch.matmul(temp_a, Y_train)
        solution = torch.matmul(temp_left, temp_right)
        return solution


class UnFLGC(torch.nn.Module):
    """
    un-supervised FLGC models with closed-form solution
    """
    def __init__(self, n_clusters, agg='appnp', regularization_coef=1e-5, K_hop=1, ro=0.8, save_affinity=False, **kwargs):
        super(UnFLGC, self).__init__()
        self.n_clusters = n_clusters
        self.regularization_coef = regularization_coef
        self.K_hop = K_hop
        self.ro = ro
        self.agg = agg
        if self.agg == 'sgc':
            self.sgconv = MySGConv(K=K_hop, cached=False)
        elif self.agg == 'appnp':
            if 'alpha' in kwargs:
                alpha = kwargs['alpha']
            else:
                alpha = 0.1
            self.sgconv = PageRankAgg(K=self.K_hop, alpha=alpha)
        else:
            raise Exception('no such aggregation!')
        self.save_affinity = save_affinity
        self.kwargs = kwargs

    def forward(self, data):
        """
        :param data: feed into a PyG Data that additionally holds a one-hot label matrix and a sparse adjacent matrix
        :return:
        """
        x_gconv = self.sgconv(data.x, data.edge_index, data.edge_attr)
        closed_form_solution = self.cal_closed_form_solution(x_gconv, data.x)
        self.closed_form_solution = closed_form_solution
        Coef = self.thrC(self.closed_form_solution.numpy(), self.ro)
        y_pred, C_final = self.post_proC(Coef, self.n_clusters,  4, 4)  # COIL20: 8, 18/10; ORL: 3, 1; YaleB: 4, 4
        C_ = self.closed_form_solution.numpy()
        if self.save_affinity:
            np.savez('./affinity.npz', C=C_final, C1=0.5 * (np.abs(C_) + np.abs(C_.T)))
        return y_pred

    def cal_closed_form_solution(self, x_gconv, x_node):
        """
        :param x_gconv: dense tensor, n_node * n_fea
        :param x_node: dense original node feature matrix
        :return:
        """
        x = x_gconv.transpose(1, 0)
        inv_ = torch.inverse(torch.matmul(x_gconv, x) + self.regularization_coef * torch.eye(x_gconv.shape[0]).float())
        # inv_ = torch.inverse(torch.matmul(x_gconv, X) + self.regularization_coef * adj**2)
        solution = torch.matmul(torch.matmul(inv_, x_gconv), x_node.transpose(1, 0))
        return solution

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

    def build_aff(self, C):
        N = C.shape[0]
        Cabs = np.abs(C)
        ind = np.argsort(-Cabs, 0)
        for i in range(N):
            Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
        Cksym = Cabs + Cabs.T
        return Cksym

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                      assign_labels='discretize', random_state=42)
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        return grp, L

    def cluster_accuracy(self, y_true, y_pre, return_aligned=False):
        Label1 = np.unique(y_true)
        nClass1 = len(Label1)
        Label2 = np.unique(y_pre)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            ind_cla1 = y_true == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = y_pre == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        y_best = np.zeros(y_pre.shape)
        for i in range(nClass2):
            y_best[y_pre == Label2[i]] = Label1[c[i]]

        # # calculate accuracy
        err_x = np.sum(y_true[:] != y_best[:])
        missrate = err_x.astype(float) / (y_true.shape[0])
        acc = 1. - missrate
        nmi = normalized_mutual_info_score(y_true, y_pre)
        kappa = cohen_kappa_score(y_true, y_best)
        ca = self.class_acc(y_true, y_best)
        ari = adjusted_rand_score(y_true, y_best)
        fscore = f1_score(y_true, y_best, average='micro')
        if return_aligned:
            return y_best, acc, nmi, kappa, ari, fscore, ca
        return y_best, acc, nmi, kappa, ari, fscore, ca

    def class_acc(self, y_true, y_pre):
        """
        calculate each class's acc
        :param y_true:
        :param y_pre:
        :return:
        """
        ca = []
        for c in np.unique(y_true):
            y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
            y_c_p = y_pre[np.nonzero(y_true == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        return ca
