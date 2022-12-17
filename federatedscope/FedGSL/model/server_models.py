import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from federatedscope.FedGSL.utils import *
from federatedscope.FedGSL.model.layers import *
from torch_geometric.nn import GCNConv, knn_graph
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj
from torch.autograd import Variable

logger = logging.getLogger(__name__)


class GSL(nn.Module):
    def __init__(self,
                 in_channels,
                 glob_gnn_outsize=64,
                 gsl_gnn_hids=64,
                 k_forKNN=20,
                 generator='MLP-D',
                 dropout_GNN=0.5,
                 dropout_adj_rate=0.,  # 尝试去掉试试
                 mlp_layers=2,
                 client_sampleNum=[1500, 1500, 1500],
                 device='cuda:0'):
        super(GSL, self).__init__()
        self.sparse = False
        self.generator = generator
        self.normalization = 'sym'
        self.dropout_adj_rate = dropout_adj_rate
        self.dropout_adj = nn.Dropout(p=dropout_adj_rate)
        self.dropout = dropout_GNN
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv_dense(in_channels, gsl_gnn_hids))
        # for i in range(nlayers - 2):
        #     self.layers.append(GCNConv_dense(hidden_dim, hidden_dim))
        self.layers.append(GCNConv_dense(gsl_gnn_hids, glob_gnn_outsize))
        self.offset_list = client_sampleNum

        # TODO 将GSL参数写进配置文件
        # GSL parameter
        if self.generator == 'MLP-D':
            self.graph_generator = MLP_Diag(mlp_layers=mlp_layers, isize=in_channels, k=k_forKNN, knn_metric='cosine',
                                            non_linearity='elu', i=6, sparse=0, mlp_act='tanh')
        else:
            self.graph_generator = My_MLP(mlp_layers=mlp_layers, isize=in_channels, k=k_forKNN, knn_metric='cosine',
                                          non_linearity='elu', i=6, sparse=0, mlp_act='tanh',
                                          offset_list=self.offset_list,device=device)
        # GNN module
        self.server_GNN = GCN_Net(in_channels=in_channels, out_channels=glob_gnn_outsize, hidden=gsl_gnn_hids,
                                  max_depth=2, dropout=dropout_GNN)

    def forward(self, x):

        if self.generator == 'MLP-D':
            # Graph generate
            Adj_ = self.graph_generator(x)  # TODO 待修改
            # Adjacency Matrix Post-Processing
            Adj = (Adj_ + Adj_.T) / 2
            Adj = adj_normalize(Adj, self.normalization, self.sparse)
            Adj = self.dropout_adj(Adj)
            # self.adj = top_k_adj  # TODO 待删除，可视化学到的Adj
            for i, conv in enumerate(self.layers[:-1]):
                x = conv(x, Adj)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers[-1](x, Adj)
            return x

            # Adj = Adj.to_sparse()
            # edge_index = Adj.indices()
            # edge_weight = Adj.values()

            # To obtain the global mode embedding matrix
            # res = self.server_GNN((x, edge_index, edge_weight))
            # return res
        else:
            # Graph generate
            Adj_ = self.graph_generator(x)  # TODO 待修改
            # Adjacency Matrix Post-Processing
            Adj = (Adj_ + Adj_.T) / 2 #TODO: 待增加
            Adj = self.dropout_adj(Adj)
            Adj = adj_normalize(Adj, self.normalization, self.sparse)

            # Adj = self.dropout_adj(Adj)
            for i, conv in enumerate(self.layers[:-1]):
                x = conv(x, Adj)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers[-1](x, Adj)

            return x


    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True,
    #                                      profile_memory=True) as prof:
    # print(prof.table())

    @torch.no_grad()
    def get_adj(self):
        return self.adj


# MLP-D邻接矩阵生成器
class MLP_Diag(nn.Module):
    def __init__(self, mlp_layers, isize, k, knn_metric, non_linearity, i, sparse, mlp_act):
        super(MLP_Diag, self).__init__()
        self.i = i
        self.layers = nn.ModuleList()
        for _ in range(mlp_layers):
            self.layers.append(Diag(isize))
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = non_linearity
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = torch.tanh(h)
                elif self.mlp_act == 'elu':
                    h = F.elu(h)
        return h

    def forward(self, features):
        embeddings = self.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = torch.mm(embeddings, embeddings.t())
        similarities = top_k(similarities, self.k + 1)  # TODO:待删除
        similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
        return similarities  # TODO:待修改
        # return embeddings, similarities


class My_MLP(nn.Module):
    def __init__(self, mlp_layers, isize, k, knn_metric, non_linearity, i, sparse, mlp_act, offset_list,device):
        super(My_MLP, self).__init__()
        self.i = i
        self.layers = nn.ModuleList()
        for _ in range(mlp_layers):
            self.layers.append(Diag(isize))
        # self.layers.append(nn.Linear(isize, 64))
        # for _ in range(mlp_layers - 2):
        #     self.layers.append(nn.Linear(64, isize))
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = non_linearity
        self.sparse = sparse
        self.mlp_act = mlp_act
        self.offset_list = offset_list
        self.device=device

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def forward(self, features):
        embeddings = self.internal_forward(features)
        embeddings = F.normalize(embeddings, dim=1, p=2)
        similarities = torch.mm(embeddings, embeddings.t())
        similarities = cross_subgraph_topk(similarities, self.k + 1, self.offset_list,self.device)  # TODO:待删除
        similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
        return similarities  # TODO:待修改


class GCNConv_dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output


class GCN_Net(torch.nn.Module):
    r""" GCN model from the "Semi-supervised Classification with Graph
    Convolutional Networks" paper, in ICLR'17.

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.
        max_depth (int): layers of GNN, default=2.
        dropout (float): dropout ratio, default=.0.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(GCN_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden))
            elif (i + 1) == max_depth:
                self.convs.append(GCNConv(hidden, out_channels))
            else:
                self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index, edge_weight = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x
