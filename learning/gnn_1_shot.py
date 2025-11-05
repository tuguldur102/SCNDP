import torch
import networkx as nx
from .utils import load_single_graph_as_data
from typing import Any

def gnn_1_shot_predict(
    model: Any,
    G: nx.Graph,
    K: int,
    device: Any,
  ):  
  """
    Choose K nodes, re-running the model after each deletion in 1-shot.
  """
  data = load_single_graph_as_data(G.copy()).to(device)

  with torch.no_grad():
    scores = model(data.x, data.edge_index, data.edge_prob)

  topk_nodes = scores.topk(K, largest=True).indices.tolist()
  
  return set(topk_nodes)

  
