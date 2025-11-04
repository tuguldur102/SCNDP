import time
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import torch
import random

from heuristics.greedy_es_variants import greedy_empty_set_celf, greedy_empty_set_celf_local_search
from heuristics.greedy_mis_variants import greedy_with_mis, greedy_with_mis_local_search, robust_greedy_mis_optimized
from heuristics.graph_centrality import remove_k_betweenness, remove_k_degree_centrality, remove_k_pagerank_nodes
from heuristics.utils import local_search, epc_mc_deleted

from learning.model import SAGEEdgeProbModel
from learning.gnn_1_shot import predict
from learning.greedy_gnn import greedy_gnn

SEED : int = 42

N_SAMPLE_EVAL = 100_000
N_SAMPLE_LS = 10_000

CHKPNT_ROOT = "/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/extension/learning/notebooks/gnn/checkpoints"
CKPT_PATH = f"{CHKPNT_ROOT}/best_model_cla_30_diff.pt"

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SAGEEdgeProbModel(in_dim=11, hidden_dim=256, heads=8,
                        dropout=0.4, aggr='mean').to(device)

model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

NODES_LST = [
  200, 
  # 300, 
  # 500, 
  # 1000
  ]

p_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

for NODES in tqdm(NODES_LST, desc="Processing nodes"):
  K = int(NODES * 0.1)

  print(f"K: {K}")

  graph_models = {
    # 'ER': nx.erdos_renyi_graph(NODES, 0.0443, seed=SEED),
    'BA': nx.barabasi_albert_graph(NODES, 2,seed=SEED),
    'SW': nx.watts_strogatz_graph(NODES, 4, 0.3, seed=SEED)
  }

    
  records = []
  for name_model, G in tqdm(
    graph_models.items(), 
    desc="Processing models", 
    total=len(graph_models)):

    for p in tqdm(p_lst, desc="Processing", total=len(p_lst)):

      def fresh_graph():
        H = G.copy()
        for u, v in H.edges():
          H[u][v]['p'] = p
        return H

      # Heuristics 1: Degree-Based Centrality
      t0 = time.perf_counter()
      degree_S  = remove_k_degree_centrality(fresh_graph(), K)

      # print(set(degree_S))
      epc_degree = epc_mc_deleted(fresh_graph(), set(degree_S), N_SAMPLE_EVAL)
      t_degree  = time.perf_counter() - t0

      # Heuristics 2: Betweenness
      t0 = time.perf_counter()
      between_S  = remove_k_betweenness(fresh_graph(), K)

      epc_between = epc_mc_deleted(fresh_graph(), set(between_S), N_SAMPLE_EVAL)

      t_between  = time.perf_counter() - t0

      # Heuristics 3: PageRank node
      t0 = time.perf_counter()
      pagerank_S  = remove_k_pagerank_nodes(fresh_graph(), K)

      epc_pagerank = epc_mc_deleted(fresh_graph(), set(pagerank_S), N_SAMPLE_EVAL)

      t_pagerank  = time.perf_counter() - t0

      # heuristics 4: Greedy ES CELF
      t0 = time.perf_counter()

      greedy_S, _ = greedy_empty_set_celf(
        fresh_graph(), K, num_samples=N_SAMPLE_LS)
      epc_greedy = epc_mc_deleted(fresh_graph(), greedy_S, N_SAMPLE_EVAL)
      t_greedy_es = time.perf_counter() - t0

      t0 = time.perf_counter()
      greedy_S, _ = greedy_empty_set_celf(
        fresh_graph(), K, num_samples=N_SAMPLE_LS)
      greedy_es_S_ls = local_search(fresh_graph(), greedy_S, num_samples=N_SAMPLE_LS)
      epc_greedy_es_ls = epc_mc_deleted(fresh_graph(), greedy_es_S_ls, N_SAMPLE_EVAL)
      t_greedy_es_ls = time.perf_counter() - t0

      # heuristics 5: Greedy MIS CELF
      t0 = time.perf_counter()

      greedy_mis_S, _ = greedy_with_mis(
        fresh_graph(), K, num_trails=50, num_samples=N_SAMPLE_LS)
      epc_greedy_mis = epc_mc_deleted(fresh_graph(), greedy_mis_S, N_SAMPLE_EVAL)
      t_greedy_mis = time.perf_counter() - t0

      t0 = time.perf_counter()

      greedy_mis_S, _ = greedy_with_mis(
        fresh_graph(), K, num_trails=50, num_samples=N_SAMPLE_LS)
      greedy_mis_S_ls = local_search(fresh_graph(), greedy_mis_S, num_samples=N_SAMPLE_LS)
      epc_greedy_mis_ls = epc_mc_deleted(fresh_graph(), greedy_mis_S_ls, N_SAMPLE_EVAL)
      t_greedy_mis_ls = time.perf_counter() - t0

      # metaheuristics 1: GNN (1 shot)
      t0 = time.perf_counter()
      gnn_S = predict(model, fresh_graph(), K, device)
      epc_gnn = epc_mc_deleted(fresh_graph(), gnn_S, N_SAMPLE_EVAL)
      t_gnn = time.perf_counter() - t0

      t0 = time.perf_counter()
      gnn_S = predict(model, fresh_graph(), K, device)
      gnn_S_ls = local_search(fresh_graph(), gnn_S, N_SAMPLE_LS)
      epc_gnn_ls = epc_mc_deleted(fresh_graph(), gnn_S_ls, N_SAMPLE_EVAL)
      t_gnn_ls = time.perf_counter() - t0

      # metaheuristics 2: Greedy-GNN
      t0 = time.perf_counter()
      greedy_gnn_S = greedy_gnn(model, fresh_graph(), K, device)
      epc_greedy_gnn = epc_mc_deleted(fresh_graph(), greedy_gnn_S, N_SAMPLE_EVAL)
      t_greedy_gnn = time.perf_counter() - t0

      t0 = time.perf_counter()
      greedy_gnn_S = greedy_gnn(model, fresh_graph(), K, device)
      greedy_gnn_S_ls = local_search(fresh_graph(), greedy_gnn_S, N_SAMPLE_LS)
      epc_greedy_gnn_ls = epc_mc_deleted(fresh_graph(), greedy_gnn_S_ls, N_SAMPLE_EVAL)
      t_greedy_gnn_ls = time.perf_counter() - t0
      
      for algo, t, epc in [
        ('Degree-based', t_degree, epc_degree),
        ('Betweenness', t_between, epc_between),
        ('PageRank', t_pagerank, epc_pagerank),

        ('Greedy', t_greedy_es, epc_greedy),
        ('Greedy + Local Search', t_greedy_es_ls, epc_greedy_es_ls),

        ('Greedy with MIS', t_greedy_mis, epc_greedy_mis),
        ('Greedy with MIS + Local Search', t_greedy_mis_ls, epc_greedy_mis_ls),

        ('GNN (1 shot)', t_gnn, epc_gnn),
        ('GNN (1 shot) + Local Search', t_gnn_ls, epc_gnn_ls),

        ('Greedy GNN', t_greedy_gnn, epc_greedy_gnn),
        ('Greedy GNN + Local Search', t_greedy_gnn_ls, epc_greedy_gnn_ls),
      ]:
        
        records.append({
          'model': name_model,
          'p': np.round(p, 1),
          'algo': algo,
          'time': t,
          'epc': epc
        })

  SAVE_PATH_ROOT = "/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/final/csv"

  df = pd.DataFrame(records)
  df.to_csv(f"{SAVE_PATH_ROOT}/Result_heuristics_{NODES}_{K}_all_large.csv", index=False)

###### DIST ######

for NODES in tqdm(NODES_LST, desc="Processing nodes"):
  K = int(NODES * 0.1)

  # graph_models = {
  #   # 'ER': nx.erdos_renyi_graph(NODES, 0.0443, seed=SEED),
  #   'BA': nx.barabasi_albert_graph(NODES, 2,seed=SEED),
  #   'SW': nx.watts_strogatz_graph(NODES, 4, 0.3, seed=SEED)
  # }

  dist_funcs = {
    'uniform': lambda: np.random.uniform(0.0, 1.0),
    'normal': lambda: np.clip(np.random.normal(0.5, 0.2), 0, 1),
    'beta': lambda: np.random.beta(2, 5),
  }

  records = []    
  for name_model, G in tqdm(
    graph_models.items(), 
    desc="Processing models", 
    total=len(graph_models)):
    
    # dist functions
    for name_dist, dist_func in tqdm(
      dist_funcs.items(),
      desc="Processing",
      total=len(dist_funcs)):

      def fresh_graph():
        """Fresh graph with new edge weights."""
        H = G.copy()
        for u, v in H.edges():
          H[u][v]['p'] = dist_func()
        return H

      # Heuristics 1: Degree-Based Centrality
      t0 = time.perf_counter()
      degree_S  = remove_k_degree_centrality(fresh_graph(), K)

      # print(set(degree_S))
      epc_degree = epc_mc_deleted(fresh_graph(), set(degree_S), N_SAMPLE_EVAL)
      t_degree  = time.perf_counter() - t0


      # Heuristics 2: Betweenness
      t0 = time.perf_counter()
      between_S  = remove_k_betweenness(fresh_graph(), K)

      epc_between = epc_mc_deleted(fresh_graph(), set(between_S), N_SAMPLE_EVAL)

      t_between  = time.perf_counter() - t0

      # Heuristics 3: PageRank node
      t0 = time.perf_counter()
      pagerank_S  = remove_k_pagerank_nodes(fresh_graph(), K)

      epc_pagerank = epc_mc_deleted(fresh_graph(), set(pagerank_S), N_SAMPLE_EVAL)

      t_pagerank  = time.perf_counter() - t0

      # heuristics 4: Greedy ES CELF
      t0 = time.perf_counter()

      greedy_S, _ = greedy_empty_set_celf(
        fresh_graph(), K, num_samples=N_SAMPLE_LS)
      epc_greedy = epc_mc_deleted(fresh_graph(), greedy_S, N_SAMPLE_EVAL)
      t_greedy_es = time.perf_counter() - t0

      t0 = time.perf_counter()
      greedy_S, _ = greedy_empty_set_celf(
        fresh_graph(), K, num_samples=N_SAMPLE_LS)
      greedy_es_S_ls = local_search(fresh_graph(), greedy_S, num_samples=N_SAMPLE_LS)
      epc_greedy_es_ls = epc_mc_deleted(fresh_graph(), greedy_es_S_ls, N_SAMPLE_EVAL)
      t_greedy_es_ls = time.perf_counter() - t0

      # heuristics 5: Greedy MIS CELF
      t0 = time.perf_counter()

      greedy_mis_S, _ = greedy_with_mis(
        fresh_graph(), K, num_trails=50, num_samples=N_SAMPLE_LS)
      epc_greedy_mis = epc_mc_deleted(fresh_graph(), greedy_mis_S, N_SAMPLE_EVAL)
      t_greedy_mis = time.perf_counter() - t0

      t0 = time.perf_counter()

      greedy_mis_S, _ = greedy_with_mis(
        fresh_graph(), K, num_trails=50, num_samples=N_SAMPLE_LS)
      greedy_mis_S_ls = local_search(fresh_graph(), greedy_mis_S, num_samples=N_SAMPLE_LS)
      epc_greedy_mis_ls = epc_mc_deleted(fresh_graph(), greedy_mis_S_ls, N_SAMPLE_EVAL)
      t_greedy_mis_ls = time.perf_counter() - t0

      # metaheuristics 1: GNN (1 shot)
      t0 = time.perf_counter()
      gnn_S = predict(model, fresh_graph(), K, device)
      epc_gnn = epc_mc_deleted(fresh_graph(), gnn_S, N_SAMPLE_EVAL)
      t_gnn = time.perf_counter() - t0

      t0 = time.perf_counter()
      gnn_S = predict(model, fresh_graph(), K, device)
      gnn_S_ls = local_search(fresh_graph(), gnn_S, N_SAMPLE_LS)
      epc_gnn_ls = epc_mc_deleted(fresh_graph(), gnn_S_ls, N_SAMPLE_EVAL)
      t_gnn_ls = time.perf_counter() - t0

      # metaheuristics 2: Greedy-GNN
      t0 = time.perf_counter()
      greedy_gnn_S = greedy_gnn(model, fresh_graph(), K, device)
      epc_greedy_gnn = epc_mc_deleted(fresh_graph(), greedy_gnn_S, N_SAMPLE_EVAL)
      t_greedy_gnn = time.perf_counter() - t0

      t0 = time.perf_counter()
      greedy_gnn_S = greedy_gnn(model, fresh_graph(), K, device)
      greedy_gnn_S_ls = local_search(fresh_graph(), greedy_gnn_S, N_SAMPLE_LS)
      epc_greedy_gnn_ls = epc_mc_deleted(fresh_graph(), greedy_gnn_S_ls, N_SAMPLE_EVAL)
      t_greedy_gnn_ls = time.perf_counter() - t0
      
      for algo, t, epc in [
        ('Degree-based', t_degree, epc_degree),
        ('Betweenness', t_between, epc_between),
        ('PageRank', t_pagerank, epc_pagerank),

        ('Greedy', t_greedy_es, epc_greedy),
        ('Greedy + Local Search', t_greedy_es_ls, epc_greedy_es_ls),

        ('Greedy with MIS', t_greedy_mis, epc_greedy_mis),
        ('Greedy with MIS + Local Search', t_greedy_mis_ls, epc_greedy_mis_ls),

        ('GNN (1 shot)', t_gnn, epc_gnn),
        ('GNN (1 shot) + Local Search', t_gnn_ls, epc_gnn_ls),

        ('Greedy GNN', t_greedy_gnn, epc_greedy_gnn),
        ('Greedy GNN + Local Search', t_greedy_gnn_ls, epc_greedy_gnn_ls),
      ]:
        
        records.append({
          'model': name_model,
          'name_dist': name_dist,
          'algo': algo,
          'time': t,
          'epc': epc
        })

  SAVE_PATH_ROOT = "/home/tuguldurb/Development/Research/SCNDP/src/SCNDP/src/final/csv"

  df = pd.DataFrame(records)
  df.to_csv(f"{SAVE_PATH_ROOT}/Result_heuristics_{NODES}_{K}_all_large_DIST.csv", index=False)

