import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class localAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(localAE, self).__init__()
        self.encoder = LinearEncoder(in_channels, out_channels)
        self.decoder = LinearEncoder(out_channels, in_channels)

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')
        loc_emb = self.encoder(x, edge_index)
        loc_reconstruct = self.decoder(loc_emb, edge_index)
        return loc_emb, loc_reconstruct


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
