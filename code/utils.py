import argparse
from collections import defaultdict
from datetime import datetime
from itertools import combinations
import graph_tool as gt
import json
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, lil_matrix
from typing import List, Tuple, Dict
import os

import hyp_sum as hs
import leman as lm
import summary_utils as sut


def load_hyperparams(path):
    # load hyperparameter values from json
    with open(path, "r") as f:
        return json.load(f)
    

def get_parser():
    # two arguments: path to default values
    # and path to overwrite and add additional
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to hyperparameter config file"
    )
    parser.add_argument(
        "--defaults", 
        type=str, 
        required=True, 
        help="Path to hyperparameter defaults config file"
    )
    return parser


def get_summary_file_name(algo_name: str,
                          dataset: str, 
                          run: int,
                          cfg: Dict):
    # returns the filename of the summary generated
    # by *algo_name* with seed *run*
    if algo_name in ['CoClusLSH', 'HyDRA']:
        return sut.get_summary_name(dataset,
                                    algo_name,
                                    cfg['algorithms'][algo_name], 
                                    cfg['r'], 
                                    cfg['b'], 
                                    cfg['min_size'],
                                    cfg['max_trials'],
                                    cfg['max_no_improvements'],
                                    run)
    if cfg['use_means'] == 'True' and algo_name in ['sc', 'sb', 'itcc', 'smcc', 'mcc']:
        return sut.get_summary_name(dataset,
                                    algo_name,
                                    run=run,
                                    r=cfg['datasets'][dataset]['means_nclusters'][0])
    return sut.get_summary_name(dataset,
                                algo_name,
                                run=run,
                                r=cfg['algorithms'][algo_name]['default_nclusters'])


def compute_hedge_dict_from_matrix(M: csr_matrix) -> Dict[int, List[int]]:
    # convert sparse matrix into list of hyperedges
    hedges = {x: list() for x in range(M.shape[1])}
    num_nodes = M.shape[0]
    for node_id in range(num_nodes):
        h_indices = M.getrow(node_id).nonzero()[1]
        for h in h_indices:
            hedges[h].append(node_id)
    return hedges


def compute_hedge_list_from_matrix(M: csr_matrix) -> List[List[int]]:
    # convert sparse matrix into list of hyperedges
    hedges = compute_hedge_dict_from_matrix(M)
    return [hedges[x] for x in range(M.shape[1])]


def compute_matrix_from_hedge_list(hedges: list[list[int]]):
    # creates the incidence matrix of this hypergraph
    vertices = set()
    for h in hedges:
        vertices.update(h)
    dim0 = len(vertices)
    assert np.array_equal(np.fromiter(vertices, int, dim0), np.arange(dim0))
    dim1 = len(hedges)
    M = lil_matrix((dim0, dim1))
    for hid, h in enumerate(hedges):
        for v in h:
            M[v, hid] += 1 # type: ignore
    return csr_matrix(M)


def create_graph_projection(supernodes: List,
                            hyperedges: List, 
                            weights: List,
                            is_directed: bool = False, 
                            is_weighted: bool = True):
    # creates the graph projection of the hypergraph
    edges = defaultdict(int)
    for idx, h in enumerate(hyperedges):
        for comb in combinations(h, 2):
            edges[comb] += weights[idx]
    edge_list = [(c[0], c[1], edges[c]) for c in edges]
    g = gt.Graph(directed=is_directed)
    g.add_vertex(len(supernodes))
    if is_weighted:
        g.add_edge_list(edge_list, eprops=[('weight', 'int')])
    else:
        g.add_edge_list(edge_list)
    return g


def create_bip_graph_projection(supernodes: List,
                                hyperedges: List, 
                                weights: List,
                                is_weighted: bool = True):
    # Create bipartite representation of the hypergraph.
    max_node_id = len(supernodes)
    edges = defaultdict(int)
    for idx, h in enumerate(hyperedges):
        for v in h:
            edges[(v, max_node_id + idx)] += weights[idx]
    edge_list = [(c[0], c[1], edges[c]) for c in edges]
    g = gt.Graph()
    g.add_vertex(len(supernodes))
    if is_weighted:
        g.add_edge_list(edge_list, eprops=[('weight', 'int')])
    else:
        g.add_edge_list(edge_list)
    return g


def create_vdict(hedges):
    # creates a dictionary
    #  node id -> set of hyperedge ids
    #             of hyperedges including the node
    vdict = defaultdict(set)
    for h in hedges:
        for v in hedges[h]:
            vdict[v].add(h)
    return vdict


def map_to_K(x: str | None):
    # get file size in Kb
    if x is None:
        return x
    if x[-1] == 'B':
        return float(x[:-1]) / 1024
    if x[-1] == 'K':
        return float(x[:-1])
    if x[-1] == 'M':
        return 1024 * float(x[:-1])
    if x[-1] == 'G':
        return 1024**2 * float(x[:-1])
    if x[-1] == 'T':
        return 1024**3 * float(x[:-1])
    return None


def compute_stats(H):
    # compute relevant statistics 
    # of hypergraph *H*
    vertices = set()
    vdegs = defaultdict(int)
    hsizes = []

    for h in H:
        vertices.update(h)
        for v in h:
            vdegs[v] += 1
        hsizes.append(len(h))

    # num nodes
    n = len(vertices)
    # num hedges
    m = len(hsizes)
    # min/mean/max hedge size
    min_d = min(hsizes)
    mean_d = np.mean(hsizes)
    max_d = max(hsizes)
    # min/mean/max node degree
    all_degs = list(vdegs.values())
    min_deg = min(all_degs)
    mean_deg = np.mean(all_degs)
    max_deg = max(all_degs)
    return {
        'Num Nodes': n, 
        'Num Edges': m, 
        'Min Hedge Size': min_d, 
        'Mean Hedge Size': np.round(mean_d, 3), 
        'Max Hedge Size': max_d,
        'Min Deg': min_deg, 
        'Mean Deg': np.round(mean_deg, 3), 
        'Max Deg': max_deg
        }


def load_dataset(file_path: str,
                 weighted: bool,
                 sep=',',
                 node_remapping=True) -> Tuple[List[List[int]], Dict[str, int], List[int]]:
    # read the hypergraph from the path *file_path*;
    # if *weighted*, we assume that the last element 
    # in each line is the weight of the hyperedge;
    # if *node_remapping*, the node ids are remapped
    # in the range [0, n-1].
    if weighted:
        end_lst = -1
    else:
        end_lst = 0
    with open(file_path) as in_f:
        H = []
        weights = []
        node_id_map = dict()
        vcount = 0
        if node_remapping:
            for line in in_f:
                lst = line.strip().split(sep)
                hedge = []
                for v in lst[:len(lst) - end_lst]:
                    v = v.strip()
                    if v not in node_id_map:
                        node_id_map[v] = vcount
                        vcount += 1
                    hedge.append(node_id_map[v]) 
                if len(hedge) > 0:
                    H.append(hedge)
                weights.append(int(lst[-1]))
        else:
            for line in in_f:
                lst = line.strip().split(sep)
                H.append([int(v) for v in lst[:len(lst) - end_lst]])
                weights.append(int(lst[-1]))
    if not weighted:
        weights = [1] * len(H)
    return H, node_id_map, weights


def load_matrix_and_hypergraph(data_path: str,
                               weighted: bool) -> Tuple[csr_matrix,
                                                        List[List[int]],
                                                        Dict[str, int],
                                                        List[int]]:
    # returns the incidence matrix and list of hyperedges
    # of the hypergraph at path *data_path*. 
    try_non_empty = True
    if data_path.endswith('.mat'):
        A = csr_matrix(scipy.io.loadmat(data_path)['A'], dtype=np.float64)
        node_id_map = dict()
        hedge_weights = list()
    elif data_path.endswith('.csv'):
        H, node_id_map, hedge_weights = load_dataset(data_path,
                                                     weighted=weighted,
                                                     node_remapping=True)
        A = compute_matrix_from_hedge_list(H)
        try_non_empty = False
    else:
        raise NotImplementedError
    if try_non_empty:
        #  non-empty rows and cols
        ne_rows = [i for i in range(A.shape[0]) if A.getrow(i).count_nonzero() > 0]
        ne_cols = [i for i in range(A.shape[1]) if A.getcol(i).count_nonzero() > 0]
        A = A[np.ix_(ne_rows, ne_cols)]
        H = compute_hedge_list_from_matrix(A)
    return A, H, node_id_map, hedge_weights


def get_biclustering(adj,
                     algo: str,
                     seed: int,
                     r: int,
                     b: int,
                     min_size: int,
                     max_trials: int,
                     max_no_improvements: int,
                     cost_thresh: float,
                     weights,
                     verbose=False):
    # Returns the summary found by Leman's algorithm
    # or by our version of the algorithm.
    if algo == 'CoClusLSH':
        return lm.hc_search(adj,
                            seed=seed, 
                            r=r,
                            b=b,
                            min_size=min_size,
                            max_trials=max_trials,
                            max_no_improvements=max_no_improvements,
                            cost_threshold=cost_thresh,
                            verbose=verbose)
    if algo == 'HyDRA':
        return hs.hc_search_hs(adj,
                               seed=seed,
                               r=r,
                               b=b,
                               min_size=min_size,
                               max_trials=max_trials,
                               cost_threshold=cost_thresh,
                               max_no_improvements=max_no_improvements,
                               weights=weights,
                               verbose=verbose)
    raise NotImplementedError


def process_stats(stats, algo: str, seed: int, dataset: str):
    # Returns the statistics of the experiment as a pandas DataFrame
    out_df = pd.DataFrame(stats, columns=['Procedure',
                                          'Time (ms)',
                                          'Delta Cost',
                                          'Round'])
    out_df['Algorithm'] = algo
    out_df['Seed'] = seed
    out_df['Dataset'] = dataset
    return out_df


def get_date_str():
    # Returns date and time as a string.
    today = str(datetime.now())
    lst = today.split(' ')
    day = lst[0]
    time = lst[1].split('.')[0]
    return f'{day}_{time}'


def generate_random_pairs(cands: List[int], size: int) -> List[Tuple[int, int]]:
    # Returns a list of random pairs of nodes.
    cand_pairs = np.random.choice(cands, (2, size))
    node_pairs = list()
    for idx in range(size):
        node_pairs.append((cand_pairs[0][idx], cand_pairs[1][idx]))
    return node_pairs


def sample_node_set_from_hypergraph(orig_path: str, 
                                    node_id_map: Dict[str, int],
                                    sample_size: int) -> Tuple[str, List[int]]:
    # Returns a list of random nodes from the hypergraph at path *orig_path*.
    file_name = os.path.basename(orig_path)
    dname = file_name.split('__')[1].split('=')[1].split('.')[0]
    nc, _, _, _, _ = sut.load_summary(orig_path, node_id_map)
    nodes = list()
    for clust in nc:
        for v in clust:
            nodes.append(v)
    return dname, list(np.random.choice(nodes, sample_size))
