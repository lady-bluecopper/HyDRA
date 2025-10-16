from collections import defaultdict
from functools import cmp_to_key
import hypergraphx as hpx
import numpy as np
import os
import subprocess
from scipy.sparse import csr_matrix
from typing import Tuple, List, Dict
import sys
sys.path.append('.')
import hyp_sum as hs
import utils as ut


def create_correction_table(node_clusters: Dict[int, List[int]],
                            hedge_clusters: Dict[int, List[int]],
                            superhyperedges: List[List[int]],
                            hedges: List[List[int]],
                            hedge_weights: List[int]):
    # create table of corrections associated to the summary
    corrections = []
    superweights_corrections = defaultdict(int)
    for cid in range(len(hedge_clusters)):
        this_corrections = []
        hedge_cluster = hedge_clusters[cid]
        for hedge in hedge_cluster:
            Nh = set(hedges[hedge])
            Nc = set()
            for supN in superhyperedges[cid]:
                Nc.update(node_clusters[supN])
            cPlus = Nh.difference(Nc)
            cMinus = Nc.difference(Nh)
            if len(cPlus) > 0 or len(cMinus) > 0:
                if len(hedge_weights) > 0 and hedge_weights[hedge] > 1:
                    this_corrections.append((cPlus, cMinus, hedge_weights[hedge]))
                else:
                    this_corrections.append((cPlus, cMinus))
            elif len(hedge_weights) > 0 and hedge_weights[hedge] > 1:
                # hedge has already be counted as 1 in the weight of the superedge
                # no corrections associated with this hedge, so we need to store its
                # weight in the superedge weight
                superweights_corrections[cid] += (hedge_weights[hedge] - 1)
        corrections.append(this_corrections)
    return corrections, superweights_corrections


def process_node(x: str, node_id_map: Dict[str, int]) -> List[int]:
    # Function used when reading a summary from disk
    # to process a supernode.
    node_lst = list()
    for y in x.strip().split(' '):
        if y not in node_id_map:
            node_id_map[y] = len(node_id_map)
        node_lst.append(node_id_map[y])
    return node_lst


def process_edge(x: str) -> Tuple[List[int], int]:
    # Function used when reading a summary from disk
    # to process a super-hyperedge.
    lst = x.strip().split(',')
    weight = int(lst[1])
    if len(lst[0].strip()) >= 1:
        edge = [int(y) for y in lst[0].split(' ')]
    else:
        edge = []
    return edge, weight


def process_correction(x: str, node_id_map: Dict[str, int]) -> Tuple[List[int], List[int], int]:
    # Function used when reading a summary from disk
    # to process an entry of the correction table.
    lst = x.strip().split(',')
    plus = []
    minus = []
    if len(lst[0]) > 0:
        plus = []
        for y in lst[0].split(' '):
            if y not in node_id_map:
                node_id_map[y] = len(node_id_map)
            plus.append(node_id_map[y])
    if len(lst[1]) > 0:
        minus = []
        for y in lst[1].split(' '):
            if y not in node_id_map:
                node_id_map[y] = len(node_id_map)
            minus.append(node_id_map[y])
    hedge_weight = 1
    if len(lst) > 2:
        hedge_weight = int(lst[2].strip())
    return plus, minus, hedge_weight


def load_summary(summary_path: str,
                 node_id_map: Dict[str, int] = dict()) -> Tuple[List[List[int]],
                                                                List[List[int]],
                                                                List[int],
                                                                Dict[int, List],
                                                                Dict[str, int]]:
    # Load summary from disk.
    supernodes = []
    superedges = []
    superedge_weights = []
    corrections = defaultdict(list)

    is_node = False
    is_edge = False
    is_corr = False

    with open(summary_path) as in_f:
        for line in in_f:
            if 'V' in line:
                is_node = True
                is_edge = False
                is_corr = False
            elif 'E' in line:
                is_node = False
                is_edge = True
                is_corr = False
            elif 'C' in line:
                is_node = False
                is_edge = False
                is_corr = True
            else:
                if is_node:
                    supernodes.append(process_node(line, node_id_map))
                elif is_edge:
                    edge, w = process_edge(line)
                    superedges.append(edge)
                    superedge_weights.append(w)
                elif is_corr:
                    if ',' not in line:
                        corr_hed_idx = int(line.strip())
                    else:
                        pl, min, hed_w = process_correction(line, node_id_map)
                        if hed_w > 1:
                            corrections[corr_hed_idx].append((pl, min, hed_w))
                        else:
                            corrections[corr_hed_idx].append((pl, min))
    return supernodes, superedges, superedge_weights, corrections, node_id_map


def sort_f(h1: List[int], h2: List[int]):
    if len(h1) > len(h2):
        return -1
    if len(h1) < len(h2):
        return 1
    if len(h1) == len(h2):
        for idx in range(len(h1)):
            if h1[idx] < h2[idx]:
                return -1
            if h1[idx] > h2[idx]:
                return 1
    return 1


cmp_key = cmp_to_key(sort_f)


def are_equal(H1: List[List[int]], H2: List[List[int]], H1_w: List[int], H2_w: List[int]) -> bool:
    # Function to determine if two hypergraphs are equivalent.
    if len(H1) != len(H2):
        return False
    if len(H1_w) > 0:
        paired_H1 = list(zip(H1, H1_w))
        paired_H2 = list(zip(H2, H2_w))
        paired_H1.sort(key=lambda x: cmp_key(x[0]))
        paired_H2.sort(key=lambda x: cmp_key(x[0]))
        H1_t, H1_w_t = zip(*paired_H1)
        H1, H1_w = list(H1_t), list(H1_w_t)
        H2_t, H2_w_t = zip(*paired_H2)
        H2, H2_w = list(H2_t), list(H2_w_t)
    else:
        H1.sort(key=cmp_key)
        H2.sort(key=cmp_key)
    for idx in range(len(H1)):
        if not np.array_equal(H1[idx], H2[idx]):
            return False
    if len(H1_w) > 0:
        return np.array_equal(H1_w, H2_w)
    return True


def get_summary_from_blocks(Nx,  # node cluster sizes
                            Ny,  # hyperedge cluster sizes
                            Qx,  # node cluster memberships
                            Qy,  # hyperedge cluster memberships
                            DnZ,  # non-zero entries in each block
                            col_cl_ids,  # remapping hedge cluster ids
                            row_cl_ids,  # remapping node cluster ids
                            alphas=[1, 1, 1],  # importance factor cost func
                            verbose=False):
    # map ids in range [0, len(node_clusters) - 1]
    n_inv_map: dict[int, int] = dict()
    for idx, clid in enumerate(row_cl_ids):
        n_inv_map[clid] = idx
    # associate ids in range [0, len(hedge_clusters) - 1]
    h_inv_map: dict[int, int] = dict()
    for idx, clid in enumerate(col_cl_ids):
        h_inv_map[clid] = idx
    # summary weights
    weights = [int(x) for x in Ny]
    # get node and hedge clusters
    rectangle_sizes = np.outer(Nx, Ny).astype(int)
    memberships = (DnZ / rectangle_sizes >= 0.5).astype(int)
    # node clusters
    node_clusters = defaultdict(list)
    for node, cid in enumerate(Qx):
        node_clusters[n_inv_map[cid]].append(node)
    # hyperedge clusters
    hedge_clusters = defaultdict(list)
    for hid, cid in enumerate(Qy):
        hedge_clusters[h_inv_map[cid]].append(hid)
    # create superedges
    num_se = len(Ny)
    superedges = [[]] * num_se
    for sh_id in range(memberships.shape[1]):
        superH = memberships.getcol(sh_id).nonzero()[0].tolist()
        superedges[sh_id] = superH
    # merge identical superedges
    consol_data = {}
    for idx, se in enumerate(superedges):
        se_tuple = tuple(sorted(se))
        if se_tuple in consol_data:
            consol_data[se_tuple]['hedge_cluster'].extend(hedge_clusters[idx])
            consol_data[se_tuple]['weight'] += weights[idx]
        else:
            consol_data[se_tuple] = {
                'superedge': se,
                'hedge_cluster': hedge_clusters[idx],
                'weight': weights[idx]
            }
    collapsed_superedges = []
    sizes = []
    collapsed_hedge_clusters = defaultdict(list)
    collapsed_weights = []
    for i, (se_tuple, data) in enumerate(consol_data.items()):
        collapsed_superedges.append(data['superedge'])
        sizes.append(len(data['superedge']))
        collapsed_hedge_clusters[i] = data['hedge_cluster']
        collapsed_weights.append(data['weight'])
    # compute cost of summary
    if verbose:
        print('num node clusters', len(node_clusters))
        print('num edge clusters', len(collapsed_hedge_clusters))
        print('num superedges', len(collapsed_superedges))
        print('sum sizes', sum(sizes))
        print('weights', collapsed_weights)
    cost = hs.hypersummary_cost(DnZ,
                                memberships,
                                rectangle_sizes,
                                sizes,
                                weights=alphas,
                                verbose=verbose)
    return node_clusters, collapsed_hedge_clusters,\
        collapsed_superedges, collapsed_weights, cost


def decode_summary(node_clusters: Dict[int, List[int]],
                   superedges: List[List[int]],
                   weights: List[int],
                   corrections: List[List],
                   return_set: bool):
    # reconstruct original hypergraph from its summary
    hypergraph = []
    hypergraph_weights = []
    for sid, superedge in enumerate(superedges):
        template = []
        for x in superedge:
            template.extend(node_clusters[x])
        tot_copies = weights[sid]
        recovered_hedges = []
        recovered_hedge_weights = []
        for correction in corrections[sid]:
            recovered = set(template)
            if len(correction[1]) > 0:
                recovered = recovered.difference(correction[1])
            if len(correction[0]) > 0:
                recovered.update(correction[0])
            rec_arr = np.array(list(recovered), dtype=np.int64)
            recovered_hedges.append(sorted(rec_arr))
            if len(correction) > 2:
                recovered_hedge_weights.append(correction[2])
            else:
                recovered_hedge_weights.append(1)
        temp_arr = sorted(template)
        temp_arr_weight = tot_copies - len(recovered_hedges)
        if temp_arr_weight > 0:
            recovered_hedges.append(temp_arr)
            recovered_hedge_weights.append(temp_arr_weight)
        hypergraph.extend(recovered_hedges)
        hypergraph_weights.extend(recovered_hedge_weights)
    # handle duplicate hyperedges!
    tmp_hyperedge_set = defaultdict(int)
    for idx, hedge in enumerate(hypergraph):
        tmp_hyperedge_set[tuple(hedge)] += hypergraph_weights[idx]
    if return_set:
        hedge_set = []
        hedge_set_weights = []
        for k, v in tmp_hyperedge_set.items():
            hedge_set.append(k)
            hedge_set_weights.append(v)
        return hedge_set, hedge_set_weights
    return hypergraph, hypergraph_weights


def get_biclustering_from_summary(out_dir: str,
                                  dataset_name: str,
                                  algo: str,
                                  cost_thresh: float,
                                  r: int,
                                  b: int,
                                  min_size: int,
                                  max_trials: int,
                                  max_no_improvements: int,
                                  run: int,
                                  node_id_map: Dict[str, int] = dict()):
    # From a summary, retrieve the node and hyperedge clusters.
    if cost_thresh == -1:
        sum_name = get_summary_name(dataset_name, algo, run=run)
    else:
        sum_name = get_summary_name(dataset_name,
                                    algo,
                                    cost_thresh,
                                    r,
                                    b,
                                    min_size,
                                    max_trials,
                                    max_no_improvements,
                                    run)
    sum_path = os.path.join(out_dir, sum_name)
    sn, se, sw, corr_dict, node_id_map = load_summary(sum_path, node_id_map)
    hcls = get_hedge_clusters_from_summary(sn, se, sw, corr_dict)

    or_name = f'original__data={dataset_name}.csv'
    or_path = os.path.join(out_dir, or_name)
    sn_or, se_or, _, _, _ = load_summary(or_path, node_id_map)
    A = ut.compute_matrix_from_hedge_list(se_or)
    H_dict = defaultdict(set)
    for hidx, hedge in enumerate(se_or):
        h_tup = tuple(sorted(hedge))
        H_dict[h_tup].add(hidx)

    Qx = np.zeros(len(sn_or))
    for s_id, supN in enumerate(sn):
        for n in supN:
            Qx[n] = s_id
    Nx = [len(supN) for supN in sn]

    Qy = np.zeros(len(se_or))
    for cl_id, h_cl in hcls.items():
        for hedge in h_cl:
            indices = list(H_dict[hedge])
            hidx = indices[0]
            H_dict[hedge].remove(hidx)
            Qy[hidx] = cl_id
    Ny = sw

    return A, Qx, Qy, Nx, Ny


def get_hpx_graph_from_summary(out_dir: str,
                               dataset_name: str,
                               algo: str,
                               cost_thresh: float,
                               r: int,
                               b: int,
                               min_size: int,
                               max_trials: int,
                               max_no_improvements: int,
                               run: int,
                               node_id_map: Dict[str, int] = dict()):
    # Create a HypergraphX hypergraph from a summary.
    if cost_thresh == -1:
        sum_name = get_summary_name(dataset_name, algo, run=run)
    else:
        sum_name = get_summary_name(dataset_name,
                                    algo,
                                    cost_thresh,
                                    r,
                                    b,
                                    min_size,
                                    max_trials,
                                    max_no_improvements,
                                    run)
    sum_path = os.path.join(out_dir, sum_name)
    _, se, sw, _, node_id_map = load_summary(sum_path, node_id_map)

    superH_tup = defaultdict(int)
    for sid, x in enumerate(se):
        if len(x) == 0:
            continue
        xtup = tuple(sorted(list(x)))
        superH_tup[xtup] += sw[sid]

    _superH = []
    _superHW = []
    for xtup, w in superH_tup.items():
        _superH.append(xtup)
        _superHW.append(w)
    return hpx.Hypergraph(edge_list=_superH, weighted=True, weights=_superHW)


def get_hedge_clusters_from_summary(node_clusters: List[List[int]],
                                    superedges: List[List[int]],
                                    weights: List[int],
                                    corrections: Dict[int, List[Tuple[List[int], List[int]]]]):
    # Create the list of hyperedge clusters from a summary.
    hedge_clusters = defaultdict(list)
    for sid, superedge in enumerate(superedges):
        template = []
        for x in superedge:
            template.extend(node_clusters[x])
        tot_copies = weights[sid]
        recovered_hedges = []
        for correction in corrections.get(sid, []):
            recovered = set(template)
            if len(correction[1]) > 0:
                recovered = recovered.difference(correction[1])
            if len(correction[0]) > 0:
                recovered.update(correction[0])
            recovered_hedges.append(tuple(sorted(list(recovered))))
        temp_tup = tuple(sorted(template))
        temp_tup_weight = tot_copies - len(recovered_hedges)
        if temp_tup_weight > 0:
            recovered_hedges.append(temp_tup)
        hedge_clusters[sid] = recovered_hedges
    return hedge_clusters


def save_summary(supernodes: Dict[int, List[int]],
                 superedges: List[List[int]],
                 weights: List[int],
                 corrections: List[List],
                 file_name: str,
                 inv_node_id_map: Dict[int, str] = dict()):
    # Save summary to disk in a format compatible with the *load_summary* function.
    with open(file_name, 'w') as out_f:
        out_f.write('V\n')
        for sn_id in range(len(supernodes)):
            out_f.write(' '.join([str(inv_node_id_map.get(x, x)) for x in supernodes[sn_id]]) + '\n')
        out_f.write('E\n')
        for se_id in range(len(superedges)):
            out_f.write(' '.join([str(x) for x in superedges[se_id]]))
            out_f.write(',')
            out_f.write(f'{weights[se_id]}\n')
        out_f.write('C\n')
        for idx, corrs in enumerate(corrections):
            if len(corrs) > 0:
                out_f.write(f'{idx}\n')
                for corr in corrs:
                    if len(corr[0]) > 0 or len(corr[1]) > 0:
                        out_f.write(' '.join([str(inv_node_id_map.get(x, x)) for x in corr[0]]))
                        out_f.write(',')
                        out_f.write(' '.join([str(inv_node_id_map.get(x, x)) for x in corr[1]]))
                        if len(corr) > 2:
                            out_f.write(',')
                            out_f.write(str(corr[2]))
                        out_f.write('\n')


def get_and_evaluate_summary(Nx: np.ndarray,  # sizes of node clusters
                             Ny: np.ndarray,  # sizes of hyperedge clusters
                             Qx: np.ndarray,  # node-cluster assignments
                             Qy: np.ndarray,  # hedge-cluster assignments
                             DnZ: csr_matrix,  # non-zero entries in each bi-cluster
                             col_cl_ids: np.ndarray,  # mapping col cluster ids
                             row_cl_ids: np.ndarray,  # mapping row cluster ids
                             A: csr_matrix,  # original adjacency matrix
                             dataset: str,  # dataset name
                             algo: str,  # algorithm name
                             cost_thresh: float,  # cost threshold to stop merging
                             r: int,  # param to generate hash codes
                             b: int,  # param to generate hash codes
                             min_size: int,  # min size of a candidate group to be processed
                             max_trials: int,  # max number of merges to try
                             max_no_improvements: int,  # Max merge tentatives with no improvements before exiting
                             run: int,  # seed used by the algorithm
                             out_dir: str,  # directory to save output
                             node_id_map: Dict[str, int]=dict(),  # mapping outer - inner node ids
                             hedge_weights = list(), # weights of hyperedges in original hypergraph
                             alphas: List[float] = [1, 1, 1],  # importance factor cost func
                             verbose: bool = False,
                             debugging: bool = False):
    inv_node_id_map = dict()
    if len(node_id_map) > 0:
        inv_node_id_map = {k: v for v, k in node_id_map.items()}

    nc, hc, se, weights, _ = get_summary_from_blocks(Nx, Ny,
                                                     Qx, Qy,
                                                     DnZ,
                                                     col_cl_ids,
                                                     row_cl_ids,
                                                     alphas=alphas,
                                                     verbose=verbose)
    H_lst = ut.compute_hedge_list_from_matrix(A)
    # find corrections
    corrections, superweights_corrections = create_correction_table(nc, hc, se, H_lst, hedge_weights)
    # apply superedge weight corrections (needed whenever original hyperedges have weights > 1)
    for hc_id, w_corr in superweights_corrections.items():
        weights[hc_id] += w_corr
    if verbose:
        num_corr = 0
        for lst in corrections:
            for cpair in lst:
                num_corr += len(cpair[0]) + len(cpair[1])
        print('num correction in table', num_corr)
    if debugging:
        # sanity check
        # some hyperedge may appear multiple time
        H_s_weights_dict = defaultdict(int)
        for x in H_lst:
            H_s_weights_dict[tuple(sorted(x))] += 1
        H_s_weights = []
        H_s = []
        for x, v in H_s_weights_dict.items():
            H_s.append(x)
            H_s_weights.append(v)
        H_decoded, H_decoded_weights = decode_summary(nc, se, weights,
                                                      corrections, True)
        # compare hedges
        if not are_equal(H_s, H_decoded, H_s_weights, H_decoded_weights):
            out_name = get_summary_name(dataset,
                                        algo,
                                        cost_thresh,
                                        r,
                                        b,
                                        min_size,
                                        max_trials,
                                        max_no_improvements,
                                        run)
            print(out_name)
            print('len(H_s)', len(H_s))
            print('len(H_lst)', len(H_lst))
            print('len(H_decoded)', len(H_decoded))
            print('H_s_weights', H_s_weights)
            print('H_decoded_weights', H_decoded_weights)
        assert are_equal(H_s, H_decoded, H_s_weights, H_decoded_weights)
    # save output
    out_name = get_summary_name(dataset,
                                algo,
                                cost_thresh,
                                r,
                                b,
                                min_size,
                                max_trials,
                                max_no_improvements,
                                run)
    out_path = os.path.join(out_dir, out_name)
    save_summary(nc, se, weights, corrections,
                 out_path, inv_node_id_map=inv_node_id_map)
    # recompute cost considering hedge weights > 1
    _, SE, _, SC, _ = load_summary(out_path, dict())
    cost = summary_cost(SE, SC)
    # check size
    size = get_file_size(out_path)
    size_K = ut.map_to_K(size)
    # recompute cost considering hedge weights > 1
    out_name = f'original__data={dataset}.csv'
    out_path = os.path.join(out_dir, out_name)
    _, SE_or, _, SC_or, _ = load_summary(out_path, dict())
    orig_cost = summary_cost(SE_or, SC_or)
    # check original size
    orig_size = get_file_size(out_path)
    orig_size_K = ut.map_to_K(orig_size)
    out = {
        'Dataset': dataset,
        'Seed': run,
        'Num Node Clusters': len(nc),
        'Num Hedge Clusters': len(se),
        'Algorithm': algo,
        'Cost': cost,
        'Orig. Cost': orig_cost,
        'Size (K)': size_K,
        'Orig. Size (K)': orig_size_K
    }
    if cost_thresh > 0:
        out.update(
            {
                'Cost Thresh.': cost_thresh,
                'r': r,
                'b': b,
                'Min Group Size': min_size,
                'Max Trials': max_trials,
                'Max No Improvement': max_no_improvements,
            }
        )
    else:
        if b > 0:
            out['Num Init Row Clusters'] = r
            out['Num Init Col Clusters'] = b
        else:
            out['Num Init Clusters'] = r
    return out


def get_summary_name(dataset: str,
                     algo: str,
                     cost_thresh: float = -1,
                     r: int = -1,
                     b: int = -1,
                     min_size: int = -1,
                     max_trials: int = -1,
                     max_no_improvements: int = -1,
                     run: int = 0):
    # Get file_name of the summary generated by *algo*
    # with the specified parameter combination.
    name = 'summary'
    name += f'__data={dataset}'
    name += f'__algo={algo}'
    if cost_thresh > 0:
        name += f'__costth={cost_thresh}'
        name += f'__r={r}'
        name += f'__b={b}'
        name += f'__min_size={min_size}'
        name += f'__max_trials={max_trials}'
        name += f'__no_imp={max_no_improvements}'
    else:
        if b > 0:
            name += f'__nrc={r}'
            name += f'__nbc={b}'
        elif r > 0:
            name += f'__nc={r}'
    name += f'__run={run}.csv'
    return name


def get_params_from_summary_name(file_name: str):
    # Extract parameter combination from filename of summary.
    lst = file_name.split('__')[1:]
    params_dict = dict()
    for el in lst:
        param_lst = el.split('=')
        param_val = param_lst[1]
        if param_val.endswith('.csv'):
            param_val = param_val[:-4]
        params_dict[param_lst[0]] = param_val
    return params_dict


def get_file_size(out_path: str):
    # Find the size of the summary file.
    months = ['jan', 'feb', 'mar',
              'apr', 'may', 'jun',
              'jul', 'aug', 'sep',
              'oct', 'nov', 'dec']

    cmd = ['ls', '-lahrt', out_path]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        # If successful, return the captured standard output
        cmd_out = result.stdout.split(' ')
        cand_size = cmd_out[4].strip()
        if len(cand_size) > 0 and cand_size[-1] in ['B', 'K', 'M', 'G', 'T']:
            return cand_size
        try:
            float(cand_size)
            return cand_size + 'B'
        except ValueError:
            for i in range(len(cmd_out)):
                if cmd_out[- i - 1].strip().lower() in months:
                    cand_size = cmd_out[- i - 2].strip()
                    if cand_size[-1] in ['B', 'K', 'M', 'G', 'T']:
                        return cand_size
                    try:
                        float(cand_size)
                        return cand_size + 'B'
                    except Exception:
                        return ''
        except Exception:
            return ''
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        return ''


def summary_cost(superedges: List[List[int]],
                 corrections: Dict[int, List[Tuple[List[int], List[int]]]],
                 weights=[1, 1, 1],
                 verbose=False):
    # cost of storing the superhyperedge weights
    e1 = len(superedges)
    # cost of storing the superhyperedges
    e2 = sum([len(se) for se in superedges])
    # cost of corrections
    e3 = 0
    for corr_lst in corrections.values():
        e3 += sum([len(c[0]) + len(c[1]) for c in corr_lst])
    # cost of hedge weights in corrections
    e4 = 0
    for corr_lst in corrections.values():
        for c in corr_lst:
            if len(c) > 2:
                e4 += int(c[2])
    if verbose:
        print('c(a)', e1, 'c(b)', e2, 'c(c)', e3, 'c(d)', e4, 'TOT', e1 + e2 + e3 + e4)
    return e1 * weights[0] + e2 * weights[1] + e3 * weights[2] + e4 * weights[0]


def random_clustering(A: csr_matrix,
                      H: List[List[int]],
                      row_nclusters: int,
                      col_nclusters: int,
                      same_size: bool,
                      seed: int):
    # Random clustering.
    np.random.seed(seed)
    tot_rows, tot_cols = A.shape
    rcids = np.arange(row_nclusters)
    ccids = np.arange(col_nclusters)
    Nx = np.zeros(row_nclusters, dtype=np.int64)
    Ny = np.zeros(col_nclusters, dtype=np.int64)
    Qx = np.zeros(tot_rows, dtype=np.int64)
    Qy = np.zeros(tot_cols, dtype=np.int64)
    # Set cluster sizes
    if same_size:
        # Strategy 1: create clusters with similar size
        r_cl_len = int(tot_rows / row_nclusters)
        for r_cl_id in range(row_nclusters):
            Nx[r_cl_id] = r_cl_len
        residual = tot_rows - np.sum(Nx)
        Nx[-1] += residual

        c_cl_len = int(tot_cols / col_nclusters)
        for c_cl_id in range(col_nclusters):
            Ny[c_cl_id] = c_cl_len
        residual = tot_cols - np.sum(Ny)
        Ny[-1] += residual
    else:
        # Strategy 2: create clusters have random positive size
        cum_r_cl_lens = 0
        for r_cl_id in range(row_nclusters):
            ub = tot_rows - cum_r_cl_lens - row_nclusters + r_cl_id + 1
            r_cl_len = np.random.randint(1, ub)
            cum_r_cl_lens += r_cl_len
            Nx[r_cl_id] = r_cl_len
        residual = tot_rows - np.sum(Nx)
        Nx[-1] += residual

        cum_c_cl_lens = 0
        for c_cl_id in range(col_nclusters):
            ub = tot_cols - cum_c_cl_lens - col_nclusters + c_cl_id + 1
            c_cl_len = np.random.randint(1, ub)
            cum_c_cl_lens += c_cl_len
            Ny[c_cl_id] = c_cl_len
        residual = tot_cols - np.sum(Ny)
        Ny[-1] += residual
    # Initialize cluster membership arrays
    r_id = 0
    for r_c_id, r_c_len in enumerate(Nx):
        for _ in range(r_c_len):
            Qx[r_id] = r_c_id
            r_id += 1
    c_id = 0
    for c_c_id, c_c_len in enumerate(Ny):
        for _ in range(c_c_len):
            Qy[c_id] = c_c_id
            c_id += 1
    # Random cluster assignment
    np.random.shuffle(Qx)
    np.random.shuffle(Qy)
    # Create cluster maps
    r_cl_map = defaultdict(list)
    for r in range(tot_rows):
        r_cl_map[Qx[r]].append(r)
    c_cl_map = defaultdict(list)
    for c in range(tot_cols):
        c_cl_map[Qy[c]].append(c)
    # Create block matrix
    DnZ = np.zeros((row_nclusters, col_nclusters))
    for ri in range(row_nclusters):
        for ci in range(col_nclusters):
            for hix in c_cl_map[ci]:
                for v in H[hix]:
                    if v in r_cl_map[ri]:
                        DnZ[ri][ci] += 1
    DnZ_mat = csr_matrix(DnZ)
    return rcids, ccids, Nx, Ny, Qx, Qy, DnZ_mat
