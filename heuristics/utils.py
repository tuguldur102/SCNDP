from typing import Tuple, Dict, List
import numpy as np
import networkx as nx
from numba import njit, prange
import random
from tqdm import tqdm
import math
from itertools import combinations, product

def sigma_exact(
  G: nx.Graph,
  S: set,
  ) -> int:
  """
    Exact EPC evaluation by considering every pairwise connections.
  """
  edges = list(G.edges())
  total_conn = 0.0

  for state in product([0,1], repeat=len(edges)):
    p_state = 1
    Gp = nx.Graph()
    Gp.add_nodes_from(set(G.nodes())-S)

    for (e, keep) in zip(edges, state):
      p_edge = G.edges[e]['p']
      p_state *= (p_edge if keep else (1-p_edge))

      if keep and e[0] not in S and e[1] not in S:
        Gp.add_edge(*e)

    for i,j in combinations(set(G.nodes())-S, 2):
      if nx.has_path(Gp, i, j):
        total_conn += p_state

  return total_conn

def component_sampling_epc_mc(G, S, num_samples=10_000,
                              epsilon=None, delta=None, use_tqdm=False):
  """
    Theoretic bounds: compute N = N(epsilon, delta) by the theoretical bound.
    Experimentation:  Otherwise, use the N as input for sample count.
  """

  # Surviving vertex set and its size
  V_remaining = set(G.nodes()) - S
  n_rem = len(V_remaining)

  # base case
  if n_rem < 2:
    return 0.0

  if num_samples is None:
    assert epsilon is not None and delta is not None
    P_E = sum(G.edges[u, v]['p'] for u, v in G.edges())
    coeff = 4 * (math.e - 2) * math.log(2 / delta)
    num_samples = math.ceil(coeff * n_rem * (n_rem - 1) /
                            (epsilon ** 2 * P_E))

  C2 = 0
  it = tqdm(range(num_samples), desc='Component sampling',
            total=num_samples) if use_tqdm else range(num_samples)

  for _ in it:
    u = random.choice(tuple(V_remaining))

    # BFS based on edge probabilities

    visited = {u}
    queue = [u]

    while queue:

      v = queue.pop()
      for w in G.neighbors(v):

        # flip a coin biased by the edge probability
        # w not in deleted nodes
        if w in V_remaining and random.random() < G.edges[v, w]['p']:

          # if w is not visited
          if w not in visited:
              visited.add(w)
              queue.append(w)

    # component counting
    C2 += (len(visited) - 1)

  return (n_rem * C2) / (2 * num_samples)

def local_search(
  G: nx.Graph,
  S_init: set,
  num_samples: int = 10_000
):
  """
    1-swap local search from REGA implementation.
  """

  S = S_init.copy()
  nodes_not_in_set = set(G.nodes()) - S

  current_epc = epc_mc_deleted(G, S, num_samples)

  improved = True
  while improved:
    improved = False
    best_swap = None

    for u in list(S):
      for v in nodes_not_in_set:        
        
        D_new = (S - {u}) | {v}

        temp_epc = epc_mc_deleted(G, D_new, num_samples)

        if temp_epc < current_epc:
            current_epc = temp_epc
            best_swap = (u, v)
            improved = True

    if improved and best_swap:
      u, v = best_swap

      S.remove(u)
      S.add(v)
      nodes_not_in_set.remove(v)
      nodes_not_in_set.add(u)
  
  return S
  
def nx_to_csr(
  G: nx.Graph
  ) -> Tuple[List[int], Dict[int, int], np.ndarray, np.ndarray, np.ndarray]:

  """
    Convert an undirected graph to CSR arrays.
  """

  nodes: List[int] = list(G.nodes())
  idx_of: Dict[int, int] = {u: i for i, u in enumerate(nodes)}

  indptr: List[int] = [0]
  indices: List[int] = []
  probs: List[float] = []

  for u in nodes:
    for v in G.neighbors(u):
      indices.append(idx_of[v])
      probs.append(G.edges[u, v]['p'])
    indptr.append(len(indices))

  return (
    nodes,
    idx_of,
    np.asarray(indptr, dtype=np.int32),
    np.asarray(indices, dtype=np.int32),
    np.asarray(probs, dtype=np.float32),
    )

@njit(inline="always")
def _bfs_component_size(
    start: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    probs: np.ndarray,
    deleted: np.ndarray
  ) -> int:
  """
    Return 1 random realisation (stack BFS).
  """
  n = deleted.size
  stack = np.empty(n, dtype=np.int32)
  visited = np.zeros(n, dtype=np.uint8)

  size = 1
  top = 0
  stack[top] = start
  top += 1
  visited[start] = 1

  while top:
    top -= 1
    v = stack[top]
    for eid in range(indptr[v], indptr[v + 1]):
      w = indices[eid]
      if deleted[w]:
        continue
      if np.random.random() >= probs[eid]:
        continue
      if visited[w]:
        continue
      visited[w] = 1
      stack[top] = w
      top += 1
      size += 1
  return size - 1

@njit(parallel=True)
def epc_mc(
    indptr: np.ndarray,
    indices: np.ndarray,
    probs: np.ndarray,
    deleted: np.ndarray,
    num_samples: int
  ) -> float:
  """
    Monte‑Carlo estimator of expected pairwise connectivity (EPC).
  """

  surv = np.where(~deleted)[0]
  m = surv.size
  if m < 2:
    return 0.0

  acc = 0.0
  for _ in prange(num_samples):
    u = surv[np.random.randint(m)]
    acc += _bfs_component_size(u, indptr, indices, probs, deleted)

  return (m * acc) / (2.0 * num_samples)

def epc_mc_deleted(
    G: nx.Graph,
    S: set,
    num_samples: int = 100_000,
  ) -> float:
  
  nodes, idx_of, indptr, indices, probs = nx_to_csr(G)
  n = len(nodes)

  deleted = np.zeros(n, dtype=np.bool_)
  for u in S:
    deleted[idx_of[u]] = True

  epc = epc_mc(indptr, indices, probs, deleted, num_samples)

  return epc