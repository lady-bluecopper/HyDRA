import argparse
import os
import numpy as np
from collections import defaultdict
import sys
sys.path.append('../code')

import utils as ut
import summary_utils as sut


parser = argparse.ArgumentParser()
parser.add_argument('-M', type=int, default=10)
parser.add_argument('-p', type=int, default=10)
parser.add_argument('-r', type=int, default=10)
parser.add_argument('-e', type=float, default=0.0)
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-out_dir', type=str, default='../data')


if __name__ == '__main__':

    args = vars(parser.parse_args())
    M = args['M']  # num superedges
    p = args['p']  # supernode size
    r = args['r']  # hedge cluster size
    e = int(p * args['e'])
    ef = args['e']
    seed = args['seed']
    out_dir = args['out_dir']

    print(args)

    np.random.seed(seed)

    # GENERATE SUPEREDGES
    num_rounds = int(np.ceil(M / 2)) - 1
    max_q = sum(M - i for i in range(num_rounds))
    num_supN_req = max_q * 2
    for q in range(M, max_q + 1, 10):

        superedges = defaultdict(list)
        supN_id = 0
        current_size = 0

        for rnd in range(num_rounds):
            skip = np.arange(rnd + 1)
            actual_it = 0
            for i in range(M - rnd):
                if current_size == q:
                    break
                for j in range(M):
                    if j not in skip:
                        superedges[j].append(supN_id)
                skip += 1
                supN_id += 1
                current_size += 1
                actual_it += 1
            skip = np.arange(rnd + 1)
            for i in range(actual_it):
                for j in skip:
                    superedges[j].append(supN_id)
                skip += 1
                supN_id += 1
        supH = list(superedges.values())
        # SANITY CHECK
        v_h_dict = defaultdict(set)
        p_freqs = dict()
        for idx, h in enumerate(supH):
            for v in h:
                v_h_dict[v].add(idx)
        it_freqs = {v: len(lst) for v, lst in v_h_dict.items()}
        for v in v_h_dict:
            for u in v_h_dict:
                if v < u:
                    common_set = set(v_h_dict[v]).intersection(v_h_dict[u])
                    if len(common_set) > 0:
                        p_freqs[(v, u)] = len(common_set)
        for t in p_freqs:
            assert it_freqs[t[0]] != p_freqs[t] or it_freqs[t[1]] != p_freqs[t]
        # GENERATE SUPERNODES
        nodes = np.arange(supN_id * p)
        supnodes = [nodes[i:i + p] for i in range(0, len(nodes), p)]
        # GENERATE HYPERGRAPH
        hyperedges = []
        for supedge in supH:
            # we generate r hyperedges for each superedge
            for hid in range(r):
                # we start from the complete set of nodes in the supernodes
                # contained in the hyperedge
                hedge = set(np.array([supnodes[x] for x in supedge]).flatten())
                if e > 0:
                    # remove up to e nodes from each supernode in the superedge
                    for sid in supedge:
                        rem = np.random.randint(e)
                        to_remove = np.random.choice(supnodes[sid], rem)
                        hedge = hedge.difference(to_remove)
                    # add up to e * q nodes not already present in the supedge
                    avail_supN = list(set(np.arange(supN_id)).difference(supedge))
                    sampl_avail_supN = np.random.choice(avail_supN, len(supedge))
                    for sid in sampl_avail_supN:
                        add = np.random.randint(e)
                        to_add = np.random.choice(supnodes[sid], add)
                        hedge.update(to_add)
                hyperedges.append(hedge)
        print(f'Generated {len(hyperedges)} hyperedges')
        # Shuffle rows and cols
        A = ut.compute_matrix_from_hedge_list(hyperedges)
        shuf_idx_r = np.random.permutation(A.shape[0])
        shuf_idx_c = np.random.permutation(A.shape[1])
        A_b = A[:, shuf_idx_c]
        A_b = A_b[shuf_idx_r, :]
        H = ut.compute_hedge_list_from_matrix(A_b)
        H_w = [1 for _ in H]
        # Save Hypergraph
        fname = f'rand_hyp__M={M}_p={p}_q={q}_r={r}_e={ef}.csv'
        out_H_path = os.path.join(out_dir, fname)
        with open(out_H_path, 'w') as out_f:
            for hedge in H:
                out_f.write(','.join(str(x) for x in hedge) + '\n')
        # reindex nodes in supernodes
        old_new_r_ids = {x: i for i, x in enumerate(shuf_idx_r)}
        node_clusters = dict()
        for sidx, supN in enumerate(supnodes):
            shuf_supN = []
            for vor in supN:
                shuf_supN.append(old_new_r_ids[vor])
            node_clusters[sidx] = shuf_supN
        # membership hyperedge -> hedge cluster
        Qy = np.zeros(len(hyperedges))
        for i in range(len(supH)):
            for j in range(r):
                Qy[i * r + j] = i
        # reindex hedges in hedge clusters
        old_new_c_ids = {x: i for i, x in enumerate(shuf_idx_c)}
        hedge_clusters = defaultdict(list)
        for hid, cl in enumerate(Qy):
            hedge_clusters[cl].append(old_new_c_ids[hid])
        superedges_weights = [len(c) for c in hedge_clusters]
        corrections, sw_corrections = sut.create_correction_table(node_clusters,
                                                                  hedge_clusters,
                                                                  supH,
                                                                  H,
                                                                  H_w)
        for hc_id, w_corr in sw_corrections.items():
            superedges_weights[hc_id] += w_corr
        # Save Summary
        fname = f'summary_rand_hyp__M={M}_p={p}_q={q}_r={r}_e={ef}.csv'
        out_S_path = os.path.join(out_dir, fname)
        sut.save_summary(node_clusters,
                         supH,
                         superedges_weights,
                         corrections,
                         out_S_path)
