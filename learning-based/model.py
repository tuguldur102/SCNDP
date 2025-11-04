from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
from .edgeAwareGat import EdgeProbGATConv

class SAGEEdgeProbModel(nn.Module):
  def __init__(
    self,
    in_dim,
    hidden_dim=256,
    heads=4,
    dropout=0.3,
    aggr='lstm'
  ):
    super().__init__()

    self.conv1 = SAGEConv(in_dim, hidden_dim, aggr=aggr, normalize=True)
    self.bn1 = nn.BatchNorm1d(hidden_dim)

    self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr, normalize=True)
    self.bn2 = nn.BatchNorm1d(hidden_dim)

    self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr, normalize=True)
    self.bn3 = nn.BatchNorm1d(hidden_dim)

    self.gat_edge = EdgeProbGATConv(
      hidden_dim, hidden_dim, heads=heads, dropout=dropout
    )

    self.out = nn.Linear(heads * hidden_dim, 1)

  def forward(self, x, edge_index, edge_prob):
    h = F.relu(self.bn1(self.conv1(x, edge_index)))
    h = F.relu(self.bn2(self.conv2(h, edge_index))) + h
    h = F.relu(self.bn3(self.conv3(h, edge_index))) + h

    h = self.gat_edge(h, edge_index, edge_prob)
    return self.out(h).squeeze(-1)
