import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import numpy as np

EOS = 1e-10


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    mask = torch.zeros(raw_graph.shape, requires_grad=False, device=raw_graph.device)
    mask.scatter_(1, indices, 1)

    sparse_graph = raw_graph * mask
    # return sparse_graph, mask #TODO: 待修改，删除返回mask
    return sparse_graph


def cross_subgraph_topk(raw_graph, K, offset_list,device):
    mask = torch.zeros(raw_graph.shape,requires_grad=False,device=raw_graph.device)
    index = 0
    client_id = 0

    while index < raw_graph.shape[0]:
        b = offset_list[client_id]
        if (index + b) > (raw_graph.shape[0]):
            end = raw_graph.shape[0]
        else:
            end = index + b

        # get node index not in current subgraph
        cross_index = torch.cat((torch.arange(0, index,device=device), torch.arange(end, raw_graph.shape[0],device=device)))

        # Get the similarity between the current subgraph node and all nodes
        sub_similarities = raw_graph[index:end]

        # Get the similarity between the current subgraph node and nodes not in current subgraph
        corss_subgraph_similarities = torch.index_select(sub_similarities, 1, cross_index)


        # Get the k-nearest neighbors of the current subgraph node and other subgraph nodes
        vals, inds = corss_subgraph_similarities.topk(k=K + 1, dim=-1)  #


        # add offset
        inds = torch.where(inds >= index, inds + b, inds)


        mask[index:end].scatter_(1, inds, 1) #is equivalent to the ”mask[index:end, inds] = 1.“

        index += b

        sparse_graph = torch.mul(raw_graph, mask)

    return sparse_graph


# with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True,
#                                      profile_memory=True) as prof:
# print(prof.table())

def sort_top_k(raw_graph, K):
    values, indices = torch.sort(raw_graph, dim=-1, descending=True)
    mask = torch.zeros(raw_graph.shape)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices[:, :K]] = 1
    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def init_knn_graph(x, k_forKNN):
    adj = torch.from_numpy(nearest_neighbors(x, k_forKNN, 'cosine')).cuda()
    # adj = knn_normalize(adj, "sym")
    return adj


def nearest_neighbors(X, k, metric):
    X = X.cpu().detach().numpy()
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def knn_normalize(adj, mode, sparse=False):
    EOS = 1e-10
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def adj_normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())
