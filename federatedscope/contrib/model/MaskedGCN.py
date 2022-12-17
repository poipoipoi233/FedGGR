from federatedscope.register import register_model
import torch.nn as nn
import torch.nn.functional as F
# Build you torch or tf model class here
class MaskedGCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
        super().__init__()
        self.n_feat = n_feat  # 每个节点的特征维度 e.g., 1433 for Cora
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args

        from federatedscope.contrib.model.FEDPUB_layers import MaskedGCNConv, MaskedLinear
        self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1, args=args)
        self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1, args=args)

        if self.args.no_clsf_mask:  # 初始Flase
            self.clsif = nn.Linear(self.n_dims, self.n_clss)
        else:
            self.clsif = MaskedLinear(self.n_dims, self.n_clss, l1=l1, args=args)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x


# Instantiate your model class with config and data
def ModelBuilder(model_config, local_data):
    model = MaskedGCN(n_feat=local_data[0][1],
                  n_dims=model_config.hidden,
                  n_clss=model_config.out_channels,
                  l1=model_config.fedpub.l1,
                  args=model_config.fedpub
                  )
    return model


def call_my_net(model_config, local_data):
    if model_config.type == "gnn_MaskedGCN":
        model = ModelBuilder(model_config, local_data)
        return model

register_model("gnn_MaskedGCN", call_my_net)
