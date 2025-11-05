import torch
import networkx as nx
from typing import Any
from .dataloader import extract_node_features

def greedy_gnn(
    model: Any,
    G: nx.Graph, 
    K: int, 
    device: Any,
  ):
    """
      Iteratively pick K nodes, re-running the model after each deletion.
    """
    S = set()                          

    for _ in range(K):
        G_rel = nx.convert_node_labels_to_integers(
                    G, label_attribute='orig_id')

        x = extract_node_features(G_rel)                       # [n, 11]
        edges = list(G_rel.edges())
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ei = torch.cat([ei, ei.flip(0)], dim=1)                
        p  = torch.tensor([G_rel[u][v]['p'] for u, v in edges] * 2,
                          dtype=torch.float)

        with torch.no_grad():
            logits = model(x.to(device), ei.to(device), p.to(device)).cpu()

        v_rel  = logits.argmax().item()                        
        v_orig = G_rel.nodes[v_rel]['orig_id']                 
        S.add(v_orig)

        G.remove_node(v_orig)

    return S