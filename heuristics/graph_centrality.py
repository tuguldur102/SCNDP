import networkx as nx
from typing import Dict, Any

def remove_k_betweenness(G: nx.Graph, k: int) -> nx.Graph:
  """
    Return the k nodes with highest betweenness centrality.
  """
  bc = nx.betweenness_centrality(G)
  topk = sorted(bc, key=bc.get, reverse=True)[:k]

  return topk

def remove_k_pagerank_nodes(
  G: nx.Graph,
  k: int,
  pagerank_kwargs: Dict[str, Any] | None = None,
) -> nx.Graph:
  """
    Return the k nodes with highest PageRank centrality.
  """
  pagerank_kwargs = {} if pagerank_kwargs is None else dict(pagerank_kwargs)
  pr = nx.pagerank(G, **pagerank_kwargs)
  topk = sorted(pr, key=pr.get, reverse=True)[:k]

  return topk

def remove_k_degree_centrality(G: nx.Graph, k: int) -> nx.Graph:
  """
    Return the k nodes with highest degree centrality, 
  """
  dc = nx.degree_centrality(G)             
  topk = sorted(dc, key=dc.get, reverse=True)[:k]
  
  return topk