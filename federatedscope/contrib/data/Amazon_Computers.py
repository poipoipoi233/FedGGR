import torch
import numpy as np

from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler

from federatedscope.core.auxiliaries.splitter_builder import get_splitter
from federatedscope.core.auxiliaries.transform_builder import get_transform
import torch_geometric.transforms as T

INF = np.iinfo(np.int64).max


def load_Amazon_Computers(config):
    path = config.data.root
    name = config.data.type.lower()

    # Splitter
    splitter = get_splitter(config)

    # Transforms
    transforms_funcs = get_transform(config, 'torch_geometric')

    # Dataset
    dataset = Amazon(path, name, T.NormalizeFeatures())
    data = dataset[0]
    global_dataset = Amazon(path, name, T.NormalizeFeatures())

    # split train val test
    train_rate = 0.6
    val_rate = 0.2
    test_rate = 0.2
    num_train_per_class = int(round(train_rate * len(data.y) / dataset.num_classes))

    num_val = int(round(val_rate * len(data.y)))
    num_test = int(round(test_rate * len(data.y)))
    num_classes = dataset.num_classes

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True
    remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    data.val_mask.fill_(False)
    data.val_mask[remaining[:num_val]] = True

    data.test_mask.fill_(False)
    data.test_mask[remaining[num_val:num_val + num_test]] = True

    dataset.data = data
    global_dataset.data = data
    # random划分
    dataset = splitter(dataset[0])

    dataset = [ds for ds in dataset]
    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])

    # get local dataset
    data_dict = dict()
    for client_idx in range(1, len(dataset) + 1):
        local_data = dataset[client_idx - 1]
        # To undirected and add self-loop
        local_data.edge_index = add_self_loops(
            to_undirected(remove_self_loops(local_data.edge_index)[0]),
            num_nodes=local_data.x.shape[0])[0]
        data_dict[client_idx] = {
            'data': local_data,
            'train': [local_data],
            'val': [local_data],
            'test': [local_data]
        }
    # Keep ML split consistent with local graphs
    if global_dataset is not None:
        global_graph = global_dataset[0]
        train_mask = torch.zeros_like(global_graph.train_mask)
        val_mask = torch.zeros_like(global_graph.val_mask)
        test_mask = torch.zeros_like(global_graph.test_mask)

        for client_sampler in data_dict.values():
            if isinstance(client_sampler, Data):
                client_subgraph = client_sampler
            else:
                client_subgraph = client_sampler['data']
            train_mask[client_subgraph.index_orig[
                client_subgraph.train_mask]] = True
            val_mask[client_subgraph.index_orig[
                client_subgraph.val_mask]] = True
            test_mask[client_subgraph.index_orig[
                client_subgraph.test_mask]] = True
        global_graph.train_mask = train_mask
        global_graph.val_mask = val_mask
        global_graph.test_mask = test_mask

        data_dict[0] = {
            'data': global_graph,
            'train': [global_graph],
            'val': [global_graph],
            'test': [global_graph]
        }
    return data_dict, config

# from federatedscope.register import register_data
#
#
# def call_my_data(config):
#     if config.data.type == "computers":
#         data, modified_config = load_Amazon_Computers(config)
#         return data, modified_config
#
#
# register_data("computers", call_my_data)
