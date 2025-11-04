from typing import Tuple, Set, List
import heapq
from .utils import epc_mc_deleted, local_search
import networkx as nx

def greedy_with_mis(
  G: nx.Graph,
  k: int,
  num_trails: int = 20,
  num_samples: int = 10_000,
) -> Tuple[Set, List]:
  """ 
    Greedy algorithm for finding a set of nodes with minimal 
    expected pairwise connectivity (EPC) from 
    a maximal independent set (MIS).
  """

  best_sigma = float("inf")

  for _ in range(num_trails):
    MIS = nx.maximal_independent_set(G)
    curr_sigma = epc_mc_deleted(G, S=MIS, num_samples=num_samples)

    if curr_sigma < best_sigma:
      best_sigma = curr_sigma

  target = len(G) - k
  R = set(MIS)
  sigma_delta = []

  # Greedy grow R set until |R| = |V| - k
  while len(R) < target:
    best_j, best_sigma = None, float('inf')
    for j in G.nodes():
      if j in R:
        continue

      # delete node
      S_j = set(G.nodes()) - (R | {j})
      sigma = epc_mc_deleted(G, S=S_j, num_samples=num_samples)

      if sigma < best_sigma:
        best_sigma, best_j = sigma, j
        sigma_delta.append(best_sigma)

    R.add(best_j)

  D = set(G.nodes()) - R
  return D, sigma_delta


def greedy_with_mis_local_search(
  G: nx.Graph,
  k: int,
  num_trails: int = 20,
  num_samples: int = 10_000,
):
  """
    Greedy with local search refinement.
  """
  S = greedy_with_mis(G.copy(), k, num_trails, num_samples)
  S_opt = local_search(G.copy(), S, num_samples=num_samples)
  return S_opt


def greedy_epc_mis_celf(
  G: nx.Graph,
  k: int,
  num_trials: int = 20,
  num_samples: int = 10_000,
) -> Tuple[Set[int], List[float]]:
  """
    CELF-accelerated version of greedy_epc_mis.
  """
  best_sigma = float("inf")
  best_MIS = None

  for _ in range(num_trials):
    MIS = nx.maximal_independent_set(G)
    sigma = epc_mc_deleted(
      G.copy(), S=set(G.nodes()) - set(MIS),
      num_samples=num_samples
    )

    if sigma < best_sigma:
      best_sigma, best_MIS = sigma, MIS

  R = set(best_MIS)
  sigma_curr = best_sigma
  target = len(G) - k
  sigma_hist = [sigma_curr]

  # CELF priority queue
  pq: list[Tuple[float, int, int]] = []
  round_id = 0

  for v in G.nodes():
    if v in R:
      continue

    gain = sigma_curr - epc_mc_deleted(
      G.copy(), S=set(G.nodes()) - (R | {v}),
      num_samples=num_samples
    )
    heapq.heappush(pq, (-gain, v, round_id))

  while len(R) < target:
    neg_gain, v, last_eval = heapq.heappop(pq)

    if last_eval == round_id:
      R.add(v)
      sigma_curr += neg_gain
      sigma_hist.append(sigma_curr)
      round_id += 1
    else:
      gain = sigma_curr - epc_mc_deleted(
        G.copy(), S=set(G.nodes()) - (R | {v}),
        num_samples=num_samples
      )
      heapq.heappush(pq, (-gain, v, round_id))

  D = set(G.nodes()) - R
  return D, sigma_hist
