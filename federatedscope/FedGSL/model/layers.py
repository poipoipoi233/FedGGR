import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import numpy as np
EOS = 1e-10

class Diag(nn.Module):
    def __init__(self, input_size):
        super(Diag, self).__init__()
        self.W = nn.Parameter(torch.ones(input_size))
        self.input_size = input_size

    def forward(self, input):
        hidden = input @ torch.diag(self.W) #矩阵相乘
        return hidden

def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X.cpu(), k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj

def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj
