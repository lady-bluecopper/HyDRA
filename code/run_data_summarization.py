from coclust.coclustering import CoclustInfo as ITCC, CoclustSpecMod as SMCC, CoclustMod as MCC # type: ignore
from sklearn.cluster import SpectralCoclustering as SC, SpectralBiclustering as SB
import numpy as np
import os
import sys
import time
from typing import Dict
from tqdm.contrib.concurrent import process_map
import pandas as pd
sys.path.append('.')

import summary_utils as sut
import utils as ut
from taucc.taucc import CoClust as CC
from tautcc.tautcc import CoClust as TCC
from gahgc import GAHGC, labels_to_framework_state
from dhcc import DHCC

    
def parallel_summarization(inp):
    data_path = inp[0]
    dataset = inp[1]
    algo = inp[2]
    seed = inp[3]
    out_dir = inp[4]
    cfg = inp[5]

    weighted = True if cfg['datasets'][dataset]['weighted'] == 'True' else False
    A, H, node_id_map, hedge_weights = ut.load_matrix_and_hypergraph(data_path, weighted=weighted)
    
    start = time.time()
    if algo.startswith('Random'):
        r = cfg['datasets'][dataset]["means_nclusters"][0]
        b = cfg['datasets'][dataset]["means_nclusters"][1]
        min_size = -1
        max_trials = -1
        max_no_improvements = -1
        cost_thresh = -1
        same_size = True if algo.split('-')[1] == 'True' else False
        rcids, ccids, Nx, Ny, Qx, Qy, DnZ = sut.random_clustering(A,
                                                                  H,
                                                                  r,
                                                                  b,
                                                                  same_size,
                                                                  seed)
    elif algo in ['CoClusLSH', 'HyDRA']:
        r = cfg['r']
        b = cfg['b']
        min_size = cfg['min_size']
        max_trials = cfg['max_trials']
        max_no_improvements = cfg['max_no_improvements']
        cost_thresh = cfg['algorithms'][algo]
        _, _, Nx, Ny, Qx, Qy, DnZ, _, _, stats = ut.get_biclustering(A,
                                                                    algo,
                                                                    seed,
                                                                    r,
                                                                    b,
                                                                    min_size,
                                                                    max_trials,
                                                                    max_no_improvements,
                                                                    cost_thresh,
                                                                    cfg['weights'])
        ccids = np.unique(Qy)
        rcids = np.unique(Qx)
    else:
        cost_thresh = -1
        b = -1
        min_size = -1
        max_trials = -1
        max_no_improvements = -1
        
        use_means = (cfg['use_means'] == 'True'
                     if isinstance(cfg['use_means'], str)
                     else bool(cfg['use_means']))
        if use_means:
            n_clusters = cfg['datasets'][dataset]['means_nclusters'][0]
        else:
            try:
                n_clusters = cfg['algorithms'][algo]['default_nclusters']
            except:
                n_clusters = -1
        r = n_clusters
        
        if algo == 'cc':
            n_clusters = r = 50
            model = CC(k=50, 
                       l=50,
                       initialization='random', 
                       random_state=seed,
                       verbose=False)
        elif algo == 'tcc':
            n_clusters = r = 10
            model = TCC(k=[10, 10],
                        random_state=seed,
                        verbose=False)
        elif algo == 'sc':
            model = SC(n_clusters=n_clusters,
                       random_state=seed)
        elif algo == 'sb':
            model = SB(n_clusters=n_clusters,
                       random_state=seed)
        elif algo == 'itcc':
            if cfg['use_means']:
                nr_clusters = cfg['datasets'][dataset]['means_nclusters'][0]
                nc_clusters = cfg['datasets'][dataset]['means_nclusters'][1]
            else:
                nr_clusters = cfg['algorithms'][algo]['default_nclusters']
                nc_clusters = cfg['algorithms'][algo]['default_nclusters']
            model = ITCC(n_row_clusters=nr_clusters,
                         n_col_clusters=nc_clusters)
            r = nr_clusters
        elif algo == 'smcc':
            model = SMCC(n_clusters=n_clusters,
                         random_state=seed)
        elif algo == 'mcc':
            model = MCC(n_clusters=n_clusters,
                        random_state=seed)
        elif algo == 'gahgc':
            params = cfg['algorithms']['gahgc']
            bandwidth = params.get('bandwidth', 'median')
            model = GAHGC(
                n_clusters=n_clusters,
                alpha=params.get('alpha', 0.5),
                max_walk_length=params.get('max_walk_length', 50),
                n_walks=params.get('n_walks', 50),
                embedding_dim=params.get('embedding_dim', 32),
                window=params.get('window', 5),
                negative_samples=params.get('negative_samples', 5),
                embedding_epochs=params.get('embedding_epochs', 1),
                learning_rate=params.get('learning_rate', 0.025),
                batch_size=params.get('batch_size', 4096),
                max_iter=params.get('max_iter', 100),
                bandwidth=bandwidth,
                embedding_backend=params.get('embedding_backend', 'sgns'),
                random_state=seed,
            )
        elif algo == 'dhcc':
            params = cfg['algorithms']['dhcc']
            if use_means:
                nr_clusters = cfg['datasets'][dataset]['means_nclusters'][0]
                nc_clusters = cfg['datasets'][dataset]['means_nclusters'][1]
            else:
                nr_clusters = params.get('default_nclusters', 9)
                nc_clusters = params.get(
                    'default_n_column_clusters', nr_clusters
                )
            r = nr_clusters
            b = nc_clusters
            model = DHCC(
                n_row_clusters=nr_clusters,
                n_column_clusters=nc_clusters,
                sample_regularisation=params.get(
                    'sample_regularisation', 100.0
                ),
                feature_regularisation=params.get(
                    'feature_regularisation', 100.0
                ),
                n_neighbors=params.get('n_neighbors', 3),
                max_iter=params.get('max_iter', 100),
                tol=params.get('tol', 1e-5),
                metric=params.get('metric', 'euclidean'),
                kmeans_n_init=params.get('kmeans_n_init', 10),
                kmeans_max_iter=params.get('kmeans_max_iter', 300),
                ridge=params.get('ridge', 1e-8),
                floor=params.get('floor', 1e-12),
                n_jobs=params.get('n_jobs', None),
                random_state=seed,
            )
        else:
            raise NotImplementedError
        
        # The paper-based competitors operate directly on sparse incidence
        # matrices.  The original library competitors retain their dense
        # input contract.
        if algo in ['gahgc', 'dhcc']:
            model.fit(A)
        else:
            B = np.asarray(A.todense())
            model.fit(B)
        
        # Process output
        if algo == 'tcc':
            row_labels = model.labels_[0] # type: ignore
            col_labels = model.labels_[1] # type: ignore
        else:
            row_labels = model.row_labels_ # type: ignore
            col_labels = model.column_labels_ # type: ignore
        rcids, ccids, Nx, Ny, Qx, Qy, DnZ = labels_to_framework_state(
            A, row_labels, col_labels
        )
        
    
    end = time.time() - start
    out = sut.get_and_evaluate_summary(Nx, Ny, Qx, Qy,
                                       DnZ, ccids, rcids,
                                       A, 
                                       dataset, 
                                       algo,
                                       cost_thresh,
                                       r,
                                       b,
                                       min_size,
                                       max_trials,
                                       max_no_improvements,
                                       seed,
                                       out_dir,
                                       node_id_map=node_id_map,
                                       hedge_weights=hedge_weights)
    out['Time (s)'] = end
    out_df = pd.DataFrame.from_dict(out, orient='index').T
    # delta_df = ut.process_stats(stats, algo, seed, dataset)
    return out_df


if __name__ == '__main__':
    
    parser = ut.get_parser()
    args = parser.parse_args()
    cfg = ut.load_hyperparams(args.defaults)
    cfg_run = ut.load_hyperparams(args.config)
    cfg.update(cfg_run)

    data_dir = cfg['data_dir']
    out_dir = cfg['out_dir']
    datasets = cfg['datasets']
    dataset_names = cfg['dataset_names']
    algo_names = cfg['algorithm_names']
    algo_str = '-'.join(algo_names)
    nruns = cfg['nruns']
    workers = cfg['max_workers']

    summary_dir = os.path.join(out_dir, 'summaries')
    os.makedirs(summary_dir, exist_ok=True)
    
    for dataset_name in dataset_names:
        if dataset_name.endswith('.csv'):
            data_path = os.path.join(data_dir, dataset_name)
        else:
            data_path = os.path.join(data_dir, dataset_name, 'A.mat')
        # save original dataset in same format of summary
        weighted = True if cfg['datasets'][dataset_name]['weighted'] == 'True' else False
        A, H, n_id_map, h_weights = ut.load_matrix_and_hypergraph(data_path, weighted=weighted)
        V = {x: [x] for x in range(A.shape[0])}
        H_w = [1 for _ in range(len(H))]
        if len(h_weights) > 0:
            H_w = h_weights
        inv_node_id_map: Dict[int, str] = dict()
        if len(n_id_map) > 0:
            inv_node_id_map = {k: v for v, k in n_id_map.items()}
        out_name = f'original__data={dataset_name}.csv'
        out_path = os.path.join(summary_dir, out_name)
        sut.save_summary(V, H, H_w, list(), out_path, inv_node_id_map=inv_node_id_map)

    inputs = []
    for dataset_name in dataset_names:
        if dataset_name.endswith('.csv'):
            data_path = os.path.join(data_dir, dataset_name)
        else:
            data_path = os.path.join(data_dir, dataset_name, 'A.mat')
        for algo in algo_names:
            for seed in range(nruns):
                inputs.append([data_path,
                               dataset_name,
                               algo,
                               seed,
                               summary_dir,
                               cfg])
    
    today = ut.get_date_str()
    if workers > 1:
        outputs = process_map(parallel_summarization, inputs, max_workers=workers)
    else:
        outputs = []
        for inp in inputs:
            outputs.append(parallel_summarization(inp))
    out_df = pd.concat(outputs)
    out_fname = f'real_data__nruns={nruns}__algos={algo_str}__date={today}.csv'
    out_path = os.path.join(out_dir, out_fname)
    out_df.to_csv(out_path, index=False, header=True)
