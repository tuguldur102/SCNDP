import os, pickle, torch
from torch_geometric.data import Data
import networkx as nx
from collections import defaultdict, deque

def extract_node_features(G: nx.Graph) -> torch.ones:
  """
  Compute per-node structural features for DGL input.
  Returns: torch.FloatTensor of shape [num_nodes, 11]
  Features:
    0: degree
    1: avg neighbor degree
    2: avg neighbor clustering coeff
    3: egonet edge count
    4: egonet sum-degree minus internal edges (volume)
    5-7: l-hop neighbor sum-degree offsets for l=1,2,3
    8: voterank score
    9: eigenvector centrality
   10: k-core number
  Normalized per feature by dividing by feature-wise max.
  """
  n = G.number_of_nodes()
  feats = torch.ones(n, 11)

  # precompute degrees and clustering
  deg = dict(G.degree())
  clust = nx.clustering(G)

  # voterank ordering and scoring
  order = nx.voterank(G)
  vote_score = {u: n - i for i, u in enumerate(order)}

  # eigenvector centrality or pagerank
  eig = nx.pagerank(G, alpha=0.85)
  core = nx.core_number(G)

  # compute for each node
  for u in G.nodes():
    nbrs = list(G.neighbors(u))
    feats[u, 0] = deg[u]
    feats[u, 1] = sum(deg[v] for v in nbrs) / max(len(nbrs), 1)
    feats[u, 2] = sum(clust[v] for v in nbrs) / max(len(nbrs), 1)

    egonet = G.subgraph(nbrs + [u])
    feats[u, 3] = egonet.number_of_edges()
    feats[u, 4] = sum(deg[v] for v in egonet.nodes()) - 2 * feats[u, 3]

    # l-hop neighbor sums
    for l in (1, 2, 3):
      # BFS up to l hops
      visited = {u}
      queue = deque([(u, 0)])
      hop_nodes = set()
      while queue:
        v, d = queue.popleft()
        if d == l:
          continue
        for w in G.neighbors(v):
          if w not in visited:
            visited.add(w)
            queue.append((w, d + 1))
            if d + 1 == l:
              hop_nodes.add(w)
      feats[u, 4 + l] = sum(deg[v] - 1 for v in hop_nodes)

    feats[u, 8] = vote_score.get(u, 0)
    feats[u, 9] = eig.get(u, 0)
    feats[u, 10] = core.get(u, 0)

  # normalize each feature dimension
  for i in range(feats.size(1)):
    col = feats[:, i]
    maxval = col.max()
    if maxval > 0:
      feats[:, i] = col / maxval

  return feats


def load_single_graph_as_data(G: str) -> Data:
  """
    Load one pickled graph created by your pipeline and turn it
    into a torch_geometric.data.Data object with the same fields
    GraphEPCDataset.__getitem__ builds.
  """
  G_nx = nx.convert_node_labels_to_integers(
    G.copy(), label_attribute='orig_id'
  )

  x = extract_node_features(G_nx)

  edges = list(G_nx.edges())
  edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
  edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
  p_list = [G_nx[u][v]['p'] for u, v in edges]
  edge_prob = torch.tensor(p_list + p_list, dtype=torch.float)

  data = Data(
    x=x,
    edge_index=edge_index,
    edge_prob=edge_prob,
    idx=torch.tensor(0)
  )

  return data
