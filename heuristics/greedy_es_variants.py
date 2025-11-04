import numpy as np
from typing import Tuple, Set, Union, List
import heapq
from tqdm import tqdm
import networkx as nx
from .utils import epc_mc_deleted, nx_to_csr, local_search, epc_mc, sigma_exact, component_sampling_epc_mc

def greedy_empty_set_celf(
    G: nx.Graph,
    K: int,
    num_samples: int = 10_000
  ) -> Tuple[Set[int], List[float]]:
  """
    CELF‑accelerated greedy deletion from empty set.
  """

  S: Set[int] = set()
  sigma_curr  = epc_mc_deleted(G, S, num_samples=num_samples)
  sigma_trace = [sigma_curr]

  pq: list[Tuple[float, int, int]] = []
  round_id = 0
  for v in G.nodes():
    gain = sigma_curr - epc_mc_deleted(G, S | {v}, num_samples=num_samples)
    heapq.heappush(pq, (-gain, v, round_id))

  for _ in range(K):
    while True:
      neg_gain, v, last_eval = heapq.heappop(pq)

      if last_eval == round_id:
        S.add(v)
        sigma_curr += neg_gain      
        sigma_trace.append(sigma_curr)
        round_id += 1                
        break

      gain = sigma_curr - epc_mc_deleted(G, S | {v}, num_samples=num_samples)
      heapq.heappush(pq, (-gain, v, round_id))

      if len(S) >= K:
        break

  return S, sigma_trace

def greedy_empty_set_celf_local_search(
    G: nx.Graph,
    K: int,
    num_samples: int = 10_000
  ) -> Tuple[Set[int], List[float]]:
  """
    CELF‑accelerated greedy deletion from empty set with local search.
  """
  
  S, sigma_trace = greedy_empty_set_celf(G, K, num_samples=num_samples)
  S = local_search(G, S, num_samples=num_samples)

  return S, sigma_trace

def greedy_cndp_epc_celf(
    G: nx.Graph,
    K: int,
    num_samples: int = 10_000,
    reuse_csr: Tuple = None,
    return_trace: bool = False,
  ) -> Union[Set[int], Tuple[Set[int], List[float]]]:
  """
    Select K nodes that minimise EPC using CELF.
  """

  if reuse_csr is None:
    nodes, idx_of, indptr, indices, probs = nx_to_csr(G)
  else:
    nodes, idx_of, indptr, indices, probs = reuse_csr
  n = len(nodes)

  deleted = np.zeros(n, dtype=np.bool_)
  current_sigma = epc_mc(indptr, indices, probs, deleted, num_samples)

  pq: List[Tuple[float, int, int]] = []
  gains = np.empty(n, dtype=np.float32)

  for v in range(n):
    deleted[v] = True
    gains[v] = current_sigma - epc_mc(indptr, indices, probs, deleted, num_samples)
    deleted[v] = False
    heapq.heappush(pq, (-gains[v], v, 0))

  S: Set[int] = set()
  trace: List[float] = []
  round_ = 0

  trace.append(current_sigma)

  while len(S) < K and pq:
    neg_gain, v, last = heapq.heappop(pq)
    
    if last == round_:
      S.add(nodes[v])
      deleted[v] = True
      current_sigma += neg_gain  
      round_ += 1
      if return_trace:
        trace.append(current_sigma)
    else:
      # recompute gain lazily
      deleted[v] = True
      new_gain = current_sigma - epc_mc(indptr, indices, probs, deleted, num_samples)
      deleted[v] = False
      heapq.heappush(pq, (-new_gain, v, round_))

  return (S, trace) if return_trace else S

def optimise_epc(
  G: nx.Graph,
  K: int,
  num_samples: int = 10_000,
  return_trace: bool = False,
 ) -> Union[Set[int], Tuple[Set[int], List[float]]]:
    csr = nx_to_csr(G)
    return greedy_cndp_epc_celf(G, K, num_samples=num_samples, reuse_csr=csr, return_trace=return_trace)

def greedy_es_local_opt(
    G: nx.Graph, 
    K: int,
    num_samples: int = 10_000
  ):

  greedy_es_S = optimise_epc(
    G=G.copy(), K=K, num_samples=num_samples)

  S_opt = local_search(G.copy(), greedy_es_S, num_samples)

  return S_opt,

def greedy_cndp_epc(
    G: nx.Graph,
    K: int,
    num_samples: int = 10000,
    exact: bool = False,
    use_tqdm: bool = False
  ) -> set:
  """
    Greedy selection of S where |S| <= K
    to minimize sigma(S) via sigma_monte_carlo().
  """

  S = set()

  Sigma_delta = []
  # Current sigma(S) for the empty set
  sigma_S = 0
  if exact:
    sigma_S = sigma_exact(G, S)
  else:
    sigma_S = component_sampling_epc_mc(G, S, num_samples=num_samples)

  Sigma_delta.append(sigma_S)

  if use_tqdm:
    it = tqdm(range(K), desc='Greedy selection', total=K)
  else:
    it = range(K)

  # Greedily select K nodes
  for _ in it:
    # inits
    best_j = None
    best_gain = -float('inf')
    best_sigma = None

    # find v maximizing gain sigma(S) - sigma(S U j)
    for j in G:
      # Skip if j is already in S to avoid redundant calculations
      if j in S:
        continue

      if exact:
        sigma_Sj = sigma_exact(G, S | {j})
      else:
        sigma_Sj = component_sampling_epc_mc(G, S | {j}, num_samples=num_samples)

      gain = sigma_S - sigma_Sj

      # j <= argmax_{j in V\S} (sigma(S) - sigma(S U j))

      if gain > best_gain:
        best_gain = gain
        best_j = j
        best_sigma = sigma_Sj


    # add the best node
    if best_j is None:
      break

    S.add(best_j)
    sigma_S = best_sigma

    Sigma_delta.append(best_sigma)

  return S, Sigma_delta

