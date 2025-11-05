import os
import pickle
import glob
import torch
from torch_geometric.data import Data, Dataset
from .utils import extract_node_features

class GraphEPCDataset(Dataset):
  def __init__(self, graphs_dir: str, labels_dir: str, split:str ="train", aggr:str ='mean'):
    self.graph_paths = glob.glob(os.path.join(graphs_dir, split, '*.pkl'))
    self.labels_dir = None
    self.aggr = aggr

    if labels_dir:
      self.labels_dir = os.path.join(labels_dir, split)

  def __len__(self):
    return len(self.graph_paths)

  def __getitem__(self, idx: int):
    path = self.graph_paths[idx]
    G_nx = pickle.load(open(path, 'rb'))['graph']

    # node-level features
    x = extract_node_features(G_nx)

    # edge index & probabilities
    edges = list(G_nx.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    p_list = [G_nx[u][v]['p'] for u, v in edges]
    edge_prob = torch.tensor(p_list + p_list, dtype=torch.float)

    # labels
    lbl_name = os.path.basename(path).replace('.pkl', '_labels.pt')
    y = None
    if self.labels_dir:
      y = torch.load(os.path.join(self.labels_dir, lbl_name)).float()

    data = Data(
      x=x,
      edge_index=edge_index,
      edge_prob=edge_prob,
      y=y,
      file_name=os.path.basename(path),
      idx=torch.tensor(idx, dtype=torch.long)
    )

    return data
