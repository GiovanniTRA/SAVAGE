import torch
from torch import nn
from torch_geometric.nn import DenseGCNConv


class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, out_channels)
        self.mlp1 = nn.Sequential(nn.Linear(out_channels, out_channels * 2), nn.ReLU(),
                                  nn.Linear(out_channels * 2, out_channels))
        self.mlp2 = nn.Sequential(nn.Linear(out_channels, out_channels * 2), nn.ReLU(),
                                  nn.Linear(out_channels * 2, out_channels))

    def forward(self, x, adj):
        # chaining two convolutions with a standard relu activation

        x = self.conv1(x, adj).relu()
        return self.conv2(x, adj)

    def decode(self, z, edge_label_index):
        # cosine similarity
        return (self.mlp1(z[edge_label_index[0]]) * self.mlp2(z[edge_label_index[1]])).sum(dim=-1)

    def decode_all(self, z):
        raise NotImplementedError
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

