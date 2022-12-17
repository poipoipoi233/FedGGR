import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import torch.nn as nn
from federatedscope.register import register_model
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing, APPNP
import numpy as np
from torch.nn import Parameter
from torch.nn import Linear

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    source: https://github.com/jianhao2016/GPRGNN/blob/master/src/GNN_models.py
    '''
    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer.
            # It means where the peak at when initializing GPR weights.
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha)**np.arange(K + 1)
            TEMP[-1] = (1 - alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha)**k
        self.temp.data[-1] = (1 - self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index,
                                    edge_weight,
                                    num_nodes=x.size(0),
                                    dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class FedGSL_gpr(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 loc_gnn_outsize=64,
                 glob_gnn_outsize=64,
                 K=10,
                 ppnp='GPR_prop',
                 alpha=0.1,
                 Init='PPR',
                 Gamma=None):
        super(FedGSL_gpr, self).__init__()
        self.lin1 = Linear(in_channels, hidden)
        self.lin2 = Linear(hidden, loc_gnn_outsize)
        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)
        self.Init = Init
        self.dprate = 0.5
        self.dropout = dropout

        self.lin = nn.Linear(loc_gnn_outsize + glob_gnn_outsize, out_channels)
        # self.lins.append(nn.Linear(loc_gnn_outsize + glob_gnn_outsize, 64))
        # self.lins.append(nn.Linear(64, out_channels))

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, batch_x, edge_index, glob_emb):
        x = batch_x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)

        emb = torch.cat((x, glob_emb), dim=1)  # 直接拼接
        res = self.lin(emb)
        return res

        # for i, conv in enumerate(self.convs):
        #     x = conv(x, edge_index)
        #     if (i + 1) == len(self.convs):
        #         break
        #     x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        #
        # return res  # , loc_emb


# Instantiate your model class with config and data
def ModelBuilder(model_config, local_data):
    model = FedGSL_gpr(in_channels=local_data[0][1],
                       out_channels=model_config.out_channels,
                       hidden=model_config.fedgsl.loc_gnn_hid,
                       max_depth=model_config.fedgsl.gnn_layer,
                       dropout=model_config.dropout,
                       loc_gnn_outsize=model_config.fedgsl.loc_gnn_outsize,
                       glob_gnn_outsize=model_config.fedgsl.glob_gnn_outsize
                       )
    return model


def call_my_net(model_config, local_data):
    if model_config.type == "gnn_fedgsl_gpr":
        model = ModelBuilder(model_config, local_data)
        return model

register_model("gnn_fedgsl_gpr", call_my_net)
