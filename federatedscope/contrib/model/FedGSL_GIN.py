import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
import torch.nn as nn
from federatedscope.register import register_model
from federatedscope.core.mlp import MLP

class FedGSL_gin(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 loc_gnn_outsize=64,
                 glob_gnn_outsize=64):
        super(FedGSL_gin, self).__init__()
        self.convs = ModuleList()
        self.lins = ModuleList()

        self.num_layers = max_depth

        for i in range(max_depth):
            if i == 0:
                self.convs.append(
                    GINConv(MLP([in_channels, hidden, hidden],
                                batch_norm=True)))
            elif (i + 1) == max_depth:
                self.convs.append(
                    GINConv(
                        MLP([hidden, hidden, loc_gnn_outsize], batch_norm=True)))
            else:
                self.convs.append(
                    GINConv(MLP([hidden, hidden, hidden], batch_norm=True)))
        self.dropout = dropout



        self.lin1 = nn.Linear(loc_gnn_outsize + glob_gnn_outsize, out_channels)
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
        # loc_emb = x
        emb = torch.cat((x, glob_emb), dim=1)  # 直接拼接

        res = self.lin1(emb)
        return res  # , loc_emb


# Instantiate your model class with config and data
def ModelBuilder(model_config, local_data):
    model = FedGSL_gin(in_channels=local_data[0][1],
                       out_channels=model_config.out_channels,
                       hidden=model_config.fedgsl.loc_gnn_hid,
                       max_depth=model_config.fedgsl.gnn_layer,
                       dropout=model_config.dropout,
                       loc_gnn_outsize=model_config.fedgsl.loc_gnn_outsize,
                       glob_gnn_outsize=model_config.fedgsl.glob_gnn_outsize
                       )
    return model


def call_my_net(model_config, local_data):
    if model_config.type == "gnn_fedgsl_gin":
        model = ModelBuilder(model_config, local_data)
        return model


register_model("gnn_fedgsl_gin", call_my_net)
