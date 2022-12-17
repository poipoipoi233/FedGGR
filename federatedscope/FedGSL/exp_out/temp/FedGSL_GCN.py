import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
from federatedscope.register import register_model

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class FedGSL_GCN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 loc_gnn_outsize=64,
                 glob_gnn_outsize=64):
        super(FedGSL_GCN, self).__init__()
        self.convs = ModuleList()
        self.lins = ModuleList()
        if max_depth == 1:
            self.convs.append(GCNConv(in_channels, loc_gnn_outsize))
        else:
            for i in range(max_depth):
                if i == 0:
                    self.convs.append(GCNConv(in_channels, hidden))
                elif (i + 1) == max_depth:
                    self.convs.append(GCNConv(hidden, loc_gnn_outsize))
                else:
                    self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout

        self.attention = Attention(loc_gnn_outsize)
        self.lin1 = nn.Linear(16, out_channels)
        # self.lins.append(nn.Linear(loc_gnn_outsize + glob_gnn_outsize, 64))
        # self.lins.append(nn.Linear(64, out_channels))
    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()

    def forward(self, batch_x, edge_index, glob_emb):
        x = batch_x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))

        emb = torch.stack([x, glob_emb], dim=1)
        emb, att = self.attention(emb)
        # res = self.lins[-1](x)
        # return res
        res = self.lin1(emb)
        return res  # , loc_emb


# Instantiate your model class with config and data
def ModelBuilder(model_config, local_data):
    model = FedGSL_GCN(in_channels=local_data[0][1],
                       out_channels=model_config.out_channels,
                       hidden=model_config.fedgsl.loc_gnn_hid,
                       max_depth=model_config.fedgsl.gnn_layer,
                       dropout=model_config.dropout,
                       loc_gnn_outsize=model_config.fedgsl.loc_gnn_outsize,
                       glob_gnn_outsize=model_config.fedgsl.glob_gnn_outsize
                       )
    return model


def call_my_net(model_config, local_data):
    if model_config.type == "gnn_fedgsl_gcn":
        model = ModelBuilder(model_config, local_data)
        return model


register_model("gnn_fedgsl_gcn", call_my_net)
