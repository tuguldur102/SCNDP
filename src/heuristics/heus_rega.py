import networkx as nx
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from tqdm import tqdm

# ----- Core Sampling and Estimation -----
def sample_realization(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if random.random() < G.edges[u, v]['p']:
            H.add_edge(u, v)
    return H

def pairwise_connectivity(H: nx.Graph) -> float:
    return sum(len(c) * (len(c) - 1) / 2 for c in nx.connected_components(H))

def estimate_epc(G: nx.Graph, N: int) -> float:
    n = G.number_of_nodes()
    total = 0.0
    for _ in range(N):
        u = random.choice(list(G.nodes()))
        visited = {u}
        queue = [u]
        while queue:
            v = queue.pop(0)
            for w in G.neighbors(v):
                if w not in visited and random.random() < G.edges[v, w]['p']:
                    visited.add(w)
                    queue.append(w)
        total += (len(visited) - 1)
    return (n * total) / (2 * N)

# ----- Heuristic Removals -----
def remove_k_betweenness(G: nx.Graph, k: int) -> nx.Graph:
    bc = nx.betweenness_centrality(G)
    topk = sorted(bc, key=bc.get, reverse=True)[:k]
    H = G.copy()
    H.remove_nodes_from(topk)
    return H

def remove_k_pagerank_edges(G: nx.Graph, k: int) -> nx.Graph:
    L = nx.line_graph(G)
    pr = nx.pagerank(L)
    topk = sorted(pr, key=pr.get, reverse=True)[:k]
    H = G.copy()
    H.remove_edges_from(topk)
    return H

# ----- Sample-Average Approximation (SAA) -----
def sample_average_objective(G: nx.Graph, S: set, T: int) -> float:
    total = 0.0
    for _ in range(T):
        H = sample_realization(G)
        H.remove_nodes_from(S)
        total += pairwise_connectivity(H)
    return total / T

def greedy_initial_SAA(G: nx.Graph, k: int, T: int) -> set:
    S = set()
    candidates = set(G.nodes())
    for _ in range(k):
        best_node, best_obj = None, float('inf')
        for u in candidates:
            obj = sample_average_objective(G, S | {u}, T)
            if obj < best_obj:
                best_node, best_obj = u, obj
        S.add(best_node)
        candidates.remove(best_node)
    return S

def SAA(G: nx.Graph, k: int, T: int) -> set:
    S = greedy_initial_SAA(G, k, T)
    improved = True
    while improved:
        improved = False
        current_obj = sample_average_objective(G, S, T)
        for u in list(S):
            for v in set(G.nodes()) - S:
                newS = (S - {u}) | {v}
                new_obj = sample_average_objective(G, newS, T)
                if new_obj < current_obj:
                    S, improved, current_obj = newS, True, new_obj
                    break
            if improved:
                break
    return S

# ----- LP Relaxation for REGA -----
def solve_lp_relaxation(G: nx.Graph, D: set, k: int) -> dict:
    nodes = list(G.nodes())
    n = len(nodes)
    edges = list(G.edges())
    m = len(edges)

    idx_s = {nodes[i]: i for i in range(n)}
    idx_z = {edges[j]: n + j for j in range(m)}

    bounds = [(0,1)] * (n + m)
    for u in D:
        bounds[idx_s[u]] = (1,1)

    A_eq = np.zeros((1, n + m))
    for u in nodes:
        A_eq[0, idx_s[u]] = 1
    b_eq = [k]

    A_ub, b_ub = [], []
    for (u, v) in edges:
        iu, iv = idx_s[u], idx_s[v]
        iz = idx_z[(u, v)]
        row = np.zeros(n + m); row[iu] = 1; row[iz] = 1
        A_ub.append(row); b_ub.append(1)
        row = np.zeros(n + m); row[iv] = 1; row[iz] = 1
        A_ub.append(row); b_ub.append(1)
        row = np.zeros(n + m); row[iz] = -1; row[iu] = -1; row[iv] = -1
        A_ub.append(row); b_ub.append(-1)
    A_ub = np.array(A_ub); b_ub = np.array(b_ub)

    c = np.zeros(n + m)
    for j, (u, v) in enumerate(edges):
        c[n + j] = G.edges[u, v]['p']

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError("LP infeasible: " + res.message)
    return {nodes[i]: float(res.x[idx_s[nodes[i]]]) for i in range(n)}

def REGA_with_LP(G: nx.Graph, k: int, T_inner: int, R: int, alpha: float) -> set:
    def sample_avg(S):
        total = 0.0
        for _ in range(T_inner):
            H = sample_realization(G)
            H.remove_nodes_from(S)
            total += pairwise_connectivity(H)
        return total / T_inner

    best_S, best_obj = None, float('inf')
    nodes = set(G.nodes())

    for _ in range(R):
        D = set()
        for _ in range(k):
            s_vals = solve_lp_relaxation(G, D, k)
            rem = list(nodes - D)
            sorted_nodes = sorted(rem, key=lambda u: s_vals[u], reverse=True)
            m = max(1, int(alpha * len(sorted_nodes)))
            D.add(random.choice(sorted_nodes[:m]))
        current_obj = sample_avg(D)
        improved = True
        while improved:
            improved = False
            for u in list(D):
                for v in nodes - D:
                    newS = (D - {u}) | {v}
                    val = sample_avg(newS)
                    if val < current_obj:
                        D, current_obj, improved = newS, val, True
                        break
                if improved:
                    break
        if current_obj < best_obj:
            best_S, best_obj = D.copy(), current_obj
    return best_S

# ----- Experiment Runner including ALL algorithms -----
def run_experiments(models, ps, k,
                    T_saa, T_inner_rega, R_rega, alpha_rega, N_eval):
    records = []
    for name, G0 in tqdm(models.items(), desc="Running experiments", total=len(models)):
        for p in tqdm(ps, desc=f"Model {name} with p", total=len(ps)):
            G = G0.copy()
            for u, v in G.edges():
                G.edges[u, v]['p'] = p

            # Betweenness
            t0 = time.perf_counter()
            G_bc = remove_k_betweenness(G, k)
            t_bc = time.perf_counter() - t0

            epc_bc = estimate_epc(G_bc, N_eval)

            # PageRank
            t0 = time.perf_counter()
            G_pr = remove_k_pagerank_edges(G, k)
            t_pr = time.perf_counter() - t0

            epc_pr = estimate_epc(G_pr, N_eval)

            SAA
            t0 = time.perf_counter()
            S_saa = SAA(G, k, T_saa)
            t_saa = time.perf_counter() - t0

            G_saa = G.copy(); G_saa.remove_nodes_from(S_saa)
            epc_saa = estimate_epc(G_saa, N_eval)

            # REGA-LP
            t0 = time.perf_counter()
            S_rega = REGA_with_LP(G, k, T_inner_rega, R_rega, alpha_rega)
            t_rega = time.perf_counter() - t0

            G_rega = G.copy(); G_rega.remove_nodes_from(S_rega)
            epc_rega = estimate_epc(G_rega, N_eval)

            for algo, t, e in [
                ('Betweenness', t_bc, epc_bc),
                ('PageRank',    t_pr, epc_pr),
                ('SAA',         t_saa, epc_saa),
                ('REGA-LP',     t_rega, epc_rega)
            ]:
                records.append({
                    'model': name, 'p': p, 'algo': algo,
                    'time': t, 'epc': e
                })
    return pd.DataFrame(records)

# ----- Define models and parameters -----
models = {
    'ER': nx.gnm_random_graph(60, 120, seed=42),
    'BA': nx.barabasi_albert_graph(60, 2, seed=42),
    'SW': nx.watts_strogatz_graph(60, 4, 0.3, seed=42),
}

ps = np.arange(0.0, 1.2, 0.2)
k = 15
T_saa = 30
T_inner_rega = 1000    # inner-sample size for REGA
R_rega = 5            # restarts for REGA
alpha_rega = 0.2
N_eval = 100000       # final EPC sample count (use 1e5 for paper-quality)

# Execute experiments
df = run_experiments(models, ps, k, T_saa, T_inner_rega, R_rega, alpha_rega, N_eval)

# Plot EPC vs p and Time vs p
for name in models:
    plt.figure()
    for algo in df.algo.unique():
        sub = df[(df.model == name) & (df.algo == algo)]
        plt.plot(sub.p, sub.epc, label=algo)
    plt.title(f"{name} — EPC vs p")
    plt.xlabel("p"); plt.ylabel("EPC"); plt.grid(True); plt.legend()
    plt.savefig(f"{name}_epc_vs_p.png")

    plt.figure()
    for algo in df.algo.unique():
        sub = df[(df.model == name) & (df.algo == algo)]
        plt.plot(sub.p, sub.time, label=algo)
    plt.title(f"{name} — Time vs p")
    plt.xlabel("p"); plt.ylabel("Time (s)"); plt.grid(True); plt.legend()
    plt.savefig(f"{name}_time_vs_p.png")
plt.show()
