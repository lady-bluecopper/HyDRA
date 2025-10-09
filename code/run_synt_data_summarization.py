from glob import glob
from itertools import product # type: ignore
import numpy as np
import os
import pandas as pd
import sys
import time
from tqdm.contrib.concurrent import process_map
sys.path.append('.')

import summary_utils as sut
import utils as ut


def parallel_summarization(inp):
    data_path = inp[0]
    sum_path = inp[1]
    dname = inp[2]
    algo = inp[3]
    seed = inp[4]
    cost_thresh = inp[5]
    r = inp[6]
    b = inp[7]
    weights = inp[8]
    min_size = inp[9]
    max_trials = inp[10]
    max_no_improvements = inp[11]
    out_dir = inp[12]
    
    try:
        H, node_id_map, _ = ut.load_dataset(data_path, weighted=False)
        A = ut.compute_matrix_from_hedge_list(H)
        _, GT_e, _, GT_c, _ = sut.load_summary(sum_path, node_id_map)
        GT_cost = sut.summary_cost(GT_e, GT_c)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as exception:
        print(exception)
        return pd.DataFrame()

    start = time.time()
    _, _, Nx, Ny, Qx, Qy, DnZ, _, _, stats = ut.get_biclustering(A,
                                                                algo,
                                                                seed,
                                                                r,
                                                                b,
                                                                min_size,
                                                                max_trials,
                                                                max_no_improvements,
                                                                cost_thresh,
                                                                weights)
    ccids = np.unique(Qy)
    rcids = np.unique(Qx)
    end = time.time() - start
    out = sut.get_and_evaluate_summary(Nx, Ny, Qx, Qy,
                                       DnZ, ccids, rcids,
                                       A, dname, algo, 
                                       cost_thresh,
                                       r, b, 
                                       min_size,
                                       max_trials,
                                       max_no_improvements,
                                       seed, out_dir,
                                       node_id_map=node_id_map)
    out['Time (s)'] = end
    out['Best Cost'] = GT_cost
    out_df = pd.DataFrame.from_dict(out, orient='index').T
    delta_df = ut.process_stats(stats, algo, seed, dname)
    return out_df, delta_df


if __name__ == '__main__':
    
    parser = ut.get_parser()
    args = parser.parse_args()
    cfg = ut.load_hyperparams(args.defaults)
    cfg_run = ut.load_hyperparams(args.config)
    cfg.update(cfg_run)

    data_dir = cfg['synt_path']
    out_dir = cfg['out_dir']
    algos = cfg['algorithm_names']
    cost_threshs = [cfg['algorithms'][k] for k in algos]
    algo_str = '-'.join(algos)
    Ms = cfg['Ms']  # Number of superedges
    ps = cfg['ps']  # Node and hedge cluster size
    es = cfg['es']  # Noise level
    nruns = cfg['nruns']
    r = cfg['r']  # Signature Size
    b = cfg['b']  # Num Hash Tables
    weights = cfg['weights']
    min_size = cfg['min_size']
    max_trials = cfg['max_trials']
    max_no_improvements = cfg['max_no_improvements']
    workers = cfg['max_workers']
    
    summary_path = os.path.join(out_dir, 'summaries')

    combos = product(Ms, ps, es)
    inputs = []
    for combo in combos:
        M = combo[0]
        p = combo[1]
        r = combo[1]
        e = combo[2]
        fname = f'rand_hyp__M={M}_p={p}_q=*_r={r}_e={e}.csv'
        hyp_paths = glob(os.path.join(data_dir, fname))
        for hyp_path in hyp_paths:
            hname = os.path.basename(hyp_path)
            q = int(hname.split('_')[-3].split('=')[1])
            dname = f'M={M}_pr={p}_q={q}_e={e}'
            sname = f'summary_rand_hyp__M={M}_p={p}_q={q}_r={r}_e={e}.csv'
            sum_path = os.path.join(data_dir, sname)

            for seed in range(nruns):
                for aidx, algo in enumerate(algos):
                    inputs.append([hyp_path,
                                   sum_path,
                                   dname,
                                   algo,
                                   seed, 
                                   cost_threshs[aidx],
                                   r,
                                   b,
                                   weights,
                                   min_size,
                                   max_trials,
                                   max_no_improvements,
                                   summary_path])
    outputs = process_map(parallel_summarization, inputs, max_workers=workers)

    today = ut.get_date_str()
    out_lst = []
    for out in outputs:
        if len(out) > 0 and len(out[0]) > 0:
            out_lst.append(out[0])
    out_df = pd.concat(out_lst)
    out_df['Increase'] = (out_df['Cost'] - out_df['Best Cost']) / out_df['Cost']
    out_fname = f'synt_data__nruns={nruns}__algos={algo_str}__date={today}.csv'
    out_path = os.path.join(out_dir, out_fname)
    out_df.to_csv(out_path, index=False, header=True)

    del_lst = []
    for out in outputs:
        if len(out) > 1 and len(out[1]) > 0:
            del_lst.append(out[1])
    del_df = pd.concat(del_lst)
    del_fname = f'synt_data__delta_cost__nruns={nruns}__algos={algo_str}__date={today}.csv'
    del_path = os.path.join(out_dir, del_fname)
    del_df.to_csv(del_path, index=False, header=True)
