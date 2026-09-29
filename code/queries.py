from collections import defaultdict
import heapq
from itertools import combinations # type: ignore
import graph_tool as gt
from graph_tool.centrality import pagerank as pr
import networkx as nx
import numpy as np
import sys
import time
from typing import List, Tuple, Dict, Set
from tqdm import tqdm
sys.path.append('.')
from pagerank.program import ppr
import summary_utils as sut
import utils as ut


# PAGERANK QUERY
def compute_graph_pagerank(file_path: str,
                           is_summary: bool=False,
                           alpha: float=0.85,
                           max_iter: int=30,
                           is_directed: bool=False,
                           weighted_pr: bool=True,
                           node_id_map: Dict[str, int]={}):
    # Run PageRank on the graph projection of the hypergraph.
    sn, hedges, weights, _, _ = sut.load_summary(file_path, node_id_map)
    g = ut.create_graph_projection(sn, hedges, weights, is_directed=is_directed, is_weighted=weighted_pr)
    n = g.num_vertices()
    # initialize personalized vector
    eps = 1 / n
    pers = g.new_vertex_property("float")
    pers.a = [eps for _ in range(n)]
    # get edge weights
    ew = None
    if weighted_pr:
        ew = g.edge_properties['weight']
    # run PageRank
    st = time.time()
    PR = np.array(pr(g, damping=alpha, pers=pers, weight=ew, max_iter=max_iter).fa) # type: ignore
    out = [PR]
    if is_summary:
        ppr_nodes = dict()
        for idx, supN in enumerate(sn):
            for node in supN:
                ppr_nodes[node] = PR[idx]
        out = [[ppr_nodes[i] for i in range(len(ppr_nodes))]]
    runtime = [time.time() - st]
    return out, runtime


def compute_bip_graph_pagerank(file_path: str,
                               is_summary: bool=False,
                               alpha: float=0.85,
                               max_iter: int=30,
                               weighted_pr: bool=True,
                               node_id_map: Dict[str, int]={}):
    # Run PageRank on the bipartite representation of the hypergraph.
    sn, hedges, weights, _, _ = sut.load_summary(file_path, node_id_map)
    g = ut.create_bip_graph_projection(sn, hedges, weights, is_weighted=weighted_pr)
    n = g.num_vertices()
    # initialize personalized vector
    eps = 1 / n
    pers = g.new_vertex_property("float")
    pers.a = [eps for _ in range(n)]
    # get edge weights
    ew = None
    if weighted_pr:
        ew = g.edge_properties['weight']
    # run PageRank
    st = time.time()
    PR = np.array(pr(g, damping=alpha, pers=pers, weight=ew, max_iter=max_iter).fa) # type: ignore
    ppr_nodes = dict()
    if is_summary:
        for idx, supN in enumerate(sn):
            for node in supN:
                ppr_nodes[node] = PR[idx]
    else:
        for supN in sn:
            for node in supN:
                ppr_nodes[node] = PR[node]
    out = [[ppr_nodes[i] for i in range(len(ppr_nodes))]]
    runtime = [time.time() - st]
    return out, runtime


def compute_pagerank(file_path: str,
                     is_summary: bool=False,
                     v_init: int=-1,
                     alpha: float=0.85,
                     num_iter: int=30,
                     node_sample: List[int]=[],
                     node_id_map: Dict[str, int]={}):
    # Run PageRank on the hypergraph.
    sn, hedges, weights, _, _ = sut.load_summary(file_path, node_id_map)
    seed_set = node_sample
    if len(node_sample) == 0:
        _, seed_set = ut.sample_node_set_from_hypergraph(file_path, node_id_map, 1)
    if is_summary:
        inv_v_map = dict()
        for sid, supN in enumerate(sn):
            for v in supN:
                inv_v_map[v] = sid
        seed_set = []
        for v in node_sample:
            seed_set.append(inv_v_map[v])

    ppr_vecs = []
    runtimes = []
    for seed in seed_set:
        ppr_vec, runtime = ppr(len(sn),
                               hedges,
                               weights=weights,
                               v_init=v_init,
                               alpha=alpha,
                               seed=seed,
                               T=num_iter)
        if is_summary:
            ppr_nodes = dict()
            for idx, supN in enumerate(sn):
                for n in supN:
                    ppr_nodes[n] = ppr_vec[idx]
            ppr_upsc = [ppr_nodes[i] for i in range(len(ppr_nodes))]
            ppr_vecs.append(ppr_upsc)
        else:
            ppr_vecs.append(ppr_vec)
        runtimes.append(runtime)
    return ppr_vecs, runtimes


# DEGREE QUERY
def get_node_hyperdegrees(hyperG: List[List[int]]):
    # Returns the hyper-degrees of the nodes.
    v_adj = defaultdict(int)
    for h in hyperG:
        for v in h:
            v_adj[v] += 1
    return v_adj


def get_approx_hyperdegrees(sn, se, sw):
    # Returns the approximate hyper-degrees of the nodes.
    sup_adj = defaultdict(int)
    for idx, h in enumerate(se):
        for supN in h:
            sup_adj[supN] += sw[idx]
    output = dict()
    for idx, supN in enumerate(sn):
        for n in supN:
            output[n] = sup_adj[idx]
    return output


def get_node_degrees(hyperG: List[List[int]]):
    # Returns the degrees of the nodes.
    v_ngb = defaultdict(set)
    for h in hyperG:
        for v in h:
            v_ngb[v].update(h)
    for v in v_ngb:
        v_ngb[v].discard(v)
    return {v: len(v_ngb[v]) for v in v_ngb}


def get_approx_degrees(sn, se):
    # Returns the approximate degrees the nodes.
    sn_ngb = defaultdict(set)
    for h in se:
        for sn_idx in h:
            sn_ngb[sn_idx].update(h)
    sn_ngb_expl = defaultdict(set)
    for sn_idx in sn_ngb:
        for sn_idx2 in sn_ngb[sn_idx]:
            sn_ngb_expl[sn_idx].update(sn[sn_idx2])
    output = dict()
    for idx, supN in enumerate(sn):
        for n in supN:
            # The union contains the query node itself through its supernode;
            # graph degree excludes self-neighbors.
            output[n] = max(len(sn_ngb_expl[idx]) - 1, 0)
    return output


# REACHABILITY
def initialize_data_structures(H: List[List[int]]):
    adj = defaultdict(set)
    v_h_map = defaultdict(list)
    for hidx, h in enumerate(H):
        for v in h:
            v_h_map[v].append(hidx)
    for lst in v_h_map.values():
        for combo in combinations(lst, 2):
            adj[combo[0]].add(combo[1])
            adj[combo[1]].add(combo[0])
    return adj, v_h_map


def reachables_from_node(u, H, adj, v_h_map, max_h=np.inf) -> Dict[int, int]:
    # Returns the nodes reachable from *u* together with their distance.
    queue = []
    for h in v_h_map[u]:
        queue.append((1, h))
    heapq.heapify(queue)
    distances = {u: 0}
    visited = set()
    tot_nodes = len(v_h_map.keys())

    while len(queue) > 0 and len(visited) < tot_nodes:
        (h_i_dis, h_i) = heapq.heappop(queue)
        visited.add(h_i)
        for v in H[h_i]:
            if v not in distances:
                distances[v] = h_i_dis
        for h_j in adj[h_i]:
            if h_j not in visited and h_i_dis < max_h:
                heapq.heappush(queue, (h_i_dis + 1, h_j))
    return distances


def find_all_reachables(nodeset, H):
    # For each node in nodeset, returns the reachable
    # nodes together with their distances
    adj, v_h_map = initialize_data_structures(H)
    all_distances = dict()
    for n in nodeset:
        distances = reachables_from_node(n, H, adj, v_h_map)
        all_distances[n] = distances
    return all_distances


def are_reachable(n_pair, H, adj, v_h_map):
    # Returns true if the two nodes in *n_pair* are
    # reachable from each other.
    visited = set()
    if n_pair[0] == n_pair[1]:
        return 0
    if n_pair[0] == -1 or n_pair[1] == -1:
        return -1
    # initialization
    queue = []
    for h in v_h_map.get(n_pair[0], []):
        queue.append((1, h))
        visited.add(h)
    heapq.heapify(queue)

    while len(queue) > 0:
        (h_i_dis, h_i) = heapq.heappop(queue)
        for v in H[h_i]:
            if v == n_pair[1]:
                return h_i_dis
        for h_j in adj[h_i]:
            if h_j not in visited:
                heapq.heappush(queue, (h_i_dis + 1, h_j))
                visited.add(h_j)
    return -1


def are_reachables(node_pairs: List[Tuple[int, int]], 
                   H: List[List[int]]):
    # For each node pair in *node_pairs*, determines
    # if the two nodes in the pair are reachable from
    # each other in the original hypergraph.
    adj, v_h_map = initialize_data_structures(H)
    reachables = dict()
    for n_pair in tqdm(node_pairs):
        reachables[n_pair] = are_reachable(n_pair, H, adj, v_h_map)
    return reachables


def are_reachables_in_summary(node_pairs: List[Tuple[int, int]], 
                              node_clusters: List[List[int]], 
                              superedges: List[List[int]]):
    # For each node pair in *node_pairs*, determines
    # if the two nodes in the pair are reachable from
    # each other in the summary.
    adj, v_h_map = initialize_data_structures(superedges)
    inv_node_clust = dict()
    for cid, lst in enumerate(node_clusters):
        for v in lst:
            inv_node_clust[v] = cid
    reachables = dict()
    for n_pair in tqdm(node_pairs):
        supN_pair = (inv_node_clust.get(n_pair[0], -1), inv_node_clust.get(n_pair[1], -1))
        reachables[n_pair] = are_reachable(supN_pair, superedges, adj, v_h_map)
    return reachables


def num_connected_components(node_clusters: List[List[int]],
                             hedges: List[List[int]]):
    # Returns the number of connected components.
    edges = set()
    for h in hedges:
        for p in combinations(h, 2):
            edges.add(p)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    g_nodes = set(graph.nodes())
    for i in range(len(node_clusters)):
        if i not in g_nodes:
            graph.add_node(i)
    num_cc = 0
    for _ in nx.connected_components(graph):
        num_cc += 1
    return num_cc


def reachable_at_k(node, k, H, adj, v_h_map) -> Set[int]:
    # Returns the set of k-hop neighbors of *node*.
    visited = set()
    neigh_at_k = set()
    if node == -1:
        return neigh_at_k
    # initialization
    queue = []
    for h in v_h_map.get(node, []):
        queue.append((1, h))
        visited.add(h)
    heapq.heapify(queue)
    
    while len(queue) > 0:
        (h_i_dis, h_i) = heapq.heappop(queue)
        for v in H[h_i]:
            if v != node:
                neigh_at_k.add(v)
        for h_j in adj[h_i]:
            if h_j not in visited and h_i_dis + 1 <= k:
                heapq.heappush(queue, (h_i_dis + 1, h_j))
                visited.add(h_j)
    return neigh_at_k


def reachables_at_k(node_list: List[int],
                    k: int,
                    H: List[List[int]]) -> Dict[int, Set[int]]:
    # For each node in *node_list*, finds the set of k-hop
    # neighbors in the hypergraph.
    adj, v_h_map = initialize_data_structures(H)
    neighs_at_k = dict()
    for node in tqdm(node_list):
        neighs_at_k[node] = reachable_at_k(node, k, H, adj, v_h_map)
    return neighs_at_k


def reachables_at_k_in_summary(node_list: List[int],
                               k: int,
                               node_clusters: List[List[int]], 
                               superedges: List[List[int]]):
    # For each node in *node_list*, finds the set of k-hop
    # neighbors in the summary.
    adj, v_h_map = initialize_data_structures(superedges)
    inv_node_clust = dict()
    for cid, lst in enumerate(node_clusters):
        for v in lst:
            inv_node_clust[v] = cid

    neighs_at_k = dict()
    for node in tqdm(node_list):
        supN = inv_node_clust.get(node, -1)
        supNeighs = reachable_at_k(supN, k, superedges, adj, v_h_map)
        neighs_at_k[node] = set()
        for supNeigh in supNeighs:
            neighs_at_k[node].update(node_clusters[supNeigh])
    return neighs_at_k


# CLOSENESS CENTRALITY
def closeness_of_node(node: int,
                      num_nodes: int,
                      H: List[List[int]],
                      adj: Dict[int, Set[int]],
                      v_h_map: Dict[int, List[int]]):
    
    distances = reachables_from_node(node, H, adj, v_h_map)
    dist_sum = 0
    if len(distances) > 0:
        dist_sum = sum(distances.values())
    if dist_sum == 0:
        return 0
    closeness = (num_nodes - 1) / dist_sum
    return closeness


def closeness(node_list: List[int],
              num_nodes: int,
              H: List[List[int]]) -> Dict[int, Set[int]]:
    # Finds the closeness centrality of each node in *node_list*
    # in the hypergraph.
    adj, v_h_map = initialize_data_structures(H)
    cl_dict = dict()
    for node in tqdm(node_list):
        cl_dict[node] = closeness_of_node(node, num_nodes, H, adj, v_h_map)
    return cl_dict


def closeness_in_summary(node_list: List[int],
                         node_clusters: List[List[int]], 
                         superedges: List[List[int]]):
    # Finds the closeness centrality of each node in *node_list*
    # in the summary.
    num_nodes = len(node_clusters)
    adj, v_h_map = initialize_data_structures(superedges)
    inv_node_clust = dict()
    for cid, lst in enumerate(node_clusters):
        for v in lst:
            inv_node_clust[v] = cid

    cl_dict = dict()
    for node in tqdm(node_list):
        supN = inv_node_clust.get(node, -1)
        supN_clos = closeness_of_node(supN, num_nodes, superedges, adj, v_h_map)
        cl_dict[node] = supN_clos
    return cl_dict
