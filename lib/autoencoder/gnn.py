import torch
from torch_geometric.nn import GCNConv

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, num_features, num_hidden):
        super(GraphAutoencoder, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_hidden)
        self.conv3 = GCNConv(num_hidden, num_hidden)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def forward(self, x, pos_edge_index, neg_edge_index):
        z = self.encode(x, pos_edge_index)
        return self.decode(z, pos_edge_index, neg_edge_index)