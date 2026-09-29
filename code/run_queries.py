from collections import defaultdict
import os
import pandas as pd
import sys
import time
from tqdm.contrib.concurrent import process_map

sys.path.append('.')
import summary_utils as sut
import utils as ut
import queries as qr


def parallel_connected_components(inp):
    # Find the number of connected components of the hypergraph.
    file_path = inp[0]
    is_summary = inp[1]
    node_id_map = inp[2]

    file_name = os.path.basename(file_path)
    nc, se, _, _, _ = sut.load_summary(file_path, node_id_map)
    st = time.time()
    num_cc = qr.num_connected_components(nc, se)
    end = time.time() - st
    if is_summary:
        out = sut.get_params_from_summary_name(file_name)
    else:
        dname = file_name.split('__')[1].split('=')[1]
        out = {'data': dname}
    out['|CC|'] = num_cc
    out['Time (s)'] = end
    return out


def parallel_pagerank(inp):
    # Run the PageRank algorithm on the Hypergraph or
    # its Graph projection or Bipartite representation.
    file_path = inp[0]
    is_summary = inp[1]
    alpha = inp[2]
    num_iter = inp[3]
    node_sample = inp[4]
    version = inp[5]
    node_id_map = inp[6]

    file_name = os.path.basename(file_path)
    if version == 'Graph':
        ppr_vs, runtimes = qr.compute_graph_pagerank(file_path,
                                                     is_summary,
                                                     alpha,
                                                     num_iter,
                                                     node_id_map=node_id_map)
    elif version == 'Bipartite':
        ppr_vs, runtimes = qr.compute_bip_graph_pagerank(file_path,
                                                         is_summary,
                                                         alpha,
                                                         num_iter,
                                                         node_id_map=node_id_map)
    else:
        ppr_vs, runtimes = qr.compute_pagerank(file_path,
                                               is_summary,
                                               alpha=alpha,
                                               num_iter=num_iter,
                                               node_sample=node_sample,
                                               node_id_map=node_id_map)
    if is_summary:
        out = sut.get_params_from_summary_name(file_name)
    else:
        dname = file_name.split('__')[1].split('=')[1]
        out = {'data': dname}
    out['PPR'] = ppr_vs
    out['Alpha'] = alpha
    out['Num Iterations'] = num_iter
    out['PPR Seed Set'] = node_sample
    out['Version'] = version
    out['Time (s)'] = runtimes
    return out


def parallel_node_degrees(inp):
    # Find the (approximate) node degrees.
    file_path = inp[0]
    is_summary = inp[1]
    node_id_map = inp[2]

    file_name = os.path.basename(file_path)

    nc, se, sw, _, _ = sut.load_summary(file_path, node_id_map)
    if is_summary:
        st = time.time()
        node_degs = qr.get_approx_degrees(nc, se)
        end_1 = time.time() - st
        st = time.time()
        node_hypdegs = qr.get_approx_hyperdegrees(nc, se, sw)
        end_2 = time.time() - st

        out = sut.get_params_from_summary_name(file_name)
    else:
        st = time.time()
        node_degs = qr.get_node_degrees(se)
        end_1 = time.time() - st
        st = time.time()
        node_hypdegs = qr.get_node_hyperdegrees(se)
        end_2 = time.time() - st

        dname = file_name.split('__')[1].split('=')[1]
        out = {'data': dname}

    out['Node Degs'] = node_degs
    out['Node Hyper-Degs'] = node_hypdegs
    out['Node Degs Time (s)'] = end_1
    out['Node Hyper-Degs Time (s)'] = end_2
    return out


def parallel_reachability(inp):
    # Find the (approximate) distances between the node pairs in *node_pairs*.
    file_path = inp[0]
    is_summary = inp[1]
    node_pairs = inp[2]
    seed = inp[3]
    node_id_map = inp[4]

    file_name = os.path.basename(file_path)

    nc, se, _, _, _ = sut.load_summary(file_path, node_id_map)
    if is_summary:
        st = time.time()
        dists = qr.are_reachables_in_summary(node_pairs, nc, se)
        end = time.time() - st

        out = sut.get_params_from_summary_name(file_name)
    else:
        st = time.time()
        dists = qr.are_reachables(node_pairs, se)
        end = time.time() - st

        dname = file_name.split('__')[1].split('=')[1]
        out = {'data': dname}

    out['Reachability'] = dists
    out['Seed'] = seed
    out['Time (s)'] = end
    return out


def parallel_reachable_at_k(inp):
    # Find the (approximate) k-hop neighborhood of the nodes in *node_list*.
    file_path = inp[0]
    is_summary = inp[1]
    node_list = inp[2]
    k = inp[3]
    node_id_map = inp[4]

    file_name = os.path.basename(file_path)
    nc, se, _, _, _ = sut.load_summary(file_path, node_id_map)

    print(f'Starting {file_name} with k={k} and |Q|={len(node_list)}')
    if is_summary:
        st = time.time()
        ngb_at_k = qr.reachables_at_k_in_summary(node_list, k, nc, se)
        end = time.time() - st
        out = sut.get_params_from_summary_name(file_name)
    else:
        st = time.time()
        ngb_at_k = qr.reachables_at_k(node_list, k, se)
        end = time.time() - st
        dname = file_name.split('__')[1].split('=')[1]
        out = {'data': dname}

    out['Neighs at k'] = ngb_at_k
    out['k'] = k
    out['Time (s)'] = end
    return out


def parallel_closeness(inp):
    # Find the (approximate) closeness centrality of the nodes in *node_list*.
    file_path = inp[0]
    is_summary = inp[1]
    node_list = inp[2]
    node_id_map = inp[3]

    file_name = os.path.basename(file_path)
    nc, se, _, _, _ = sut.load_summary(file_path, node_id_map)

    print(f'Starting {file_name} with |Q|={len(node_list)}')
    if is_summary:
        st = time.time()
        cl_dict = qr.closeness_in_summary(node_list, nc, se)
        end = time.time() - st
        out = sut.get_params_from_summary_name(file_name)
    else:
        st = time.time()
        cl_dict = qr.closeness(node_list, len(nc), se)
        end = time.time() - st
        dname = file_name.split('__')[1].split('=')[1]
        out = {'data': dname}

    out['Closeness'] = cl_dict
    out['Time (s)'] = end
    return out


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
    nruns = cfg['nruns']
    workers = cfg['max_workers']
    seed = cfg['seed']

    r = cfg['r']
    b = cfg['b']
    min_size = cfg['min_size']
    max_trials = cfg['max_trials']
    max_no_improvements = cfg['max_no_improvements']

    version = cfg['version']
    alpha = cfg['alpha'][version]
    num_iter = cfg['num_iter'][version]
    sample_size = cfg['sample_size']
    query_exp = cfg['query_exp']

    summary_dir = os.path.join(out_dir, 'summaries')

    orig_paths = []  # ORIGINAL PATHS
    summary_paths = []  # SUMMARY PATHS
    orig_node_id_maps = dict()  # NODE ID MAPS
    for dataset in dataset_names:
        data_path = os.path.join(summary_dir, f'original__data={dataset}.csv')
        _, _, _, _, node_id_map = sut.load_summary(data_path, dict())
        orig_paths.append(data_path)
        dataname = dataset.split('.')[0]
        orig_node_id_maps[dataname] = node_id_map

        for algo in algo_names:
            for run in range(nruns):
                sum_name = ut.get_summary_file_name(algo, dataset, run, cfg)
                sum_path = os.path.join(summary_dir, sum_name)
                summary_paths.append(sum_path)

    # PAGERANK QUERY
    if query_exp[0]:
        node_samples = defaultdict(list)
        if version == 'Hypergraph':
            # Generate queries
            for path in orig_paths:
                dname = path.split('__')[1].split('=')[1].split('.')[0]
                dname, sampl = ut.sample_node_set_from_hypergraph(path, orig_node_id_maps[dname], sample_size)
                node_samples[dname] = sampl
        inputs = []
        for path in orig_paths:
            file_name = os.path.basename(path)
            dname = file_name.split('__')[1].split('=')[1].split('.')[0]
            inputs.append([path, False, alpha, num_iter, node_samples[dname],
                           version, orig_node_id_maps[dname]])
        for path in summary_paths:
            file_name = os.path.basename(path)
            params = sut.get_params_from_summary_name(file_name)
            dataname = params['data'].split('.')[0]
            inputs.append([path, True, alpha, num_iter,
                           node_samples[dataname],
                           version,
                           orig_node_id_maps[dataname]])
        outputs = process_map(parallel_pagerank, inputs, max_workers=workers)

        today = ut.get_date_str()
        real_ppr_path = os.path.join(out_dir, f'ACTUAL_PPR__date={today}.csv')
        approx_ppr_path = os.path.join(out_dir, f'APPROX_PPR__date={today}.csv')

        real_dfs = []
        approx_dfs = []
        for out in outputs:
            rows = []
            seed_sets = out['PPR Seed Set']
            if version != 'Hypergraph':
                seed_sets = defaultdict(list)
            runtimes = out['Time (s)']
            for sid, ppr_v in enumerate(out['PPR']):
                for v, pr in enumerate(ppr_v):
                    rows.append([v, pr, seed_sets[sid], runtimes[sid]])
            df = pd.DataFrame(rows, columns=['Node', 'PageRank', 'Seed', 'Time (s)'])
            for key in out:
                if key not in ['PPR', 'PPR Seed Set', 'Time (s)']:
                    df[key] = out[key]
            if 'algo' not in out:
                real_dfs.append(df)
            else:
                approx_dfs.append(df)
        real_df_ = pd.concat(real_dfs)
        real_df_.to_csv(real_ppr_path)
        approx_df_ = pd.concat(approx_dfs)
        approx_df_.to_csv(approx_ppr_path)

    # DEGREE QUERY
    if query_exp[1]:
        inputs = []
        for path in orig_paths:
            file_name = os.path.basename(path)
            dname = file_name.split('__')[1].split('=')[1].split('.')[0]
            inputs.append([path, False, orig_node_id_maps[dname]])
        for path in summary_paths:
            file_name = os.path.basename(path)
            params = sut.get_params_from_summary_name(file_name)
            dataname = params['data'].split('.')[0]
            inputs.append([path, True, orig_node_id_maps[dataname]])
        outputs = process_map(parallel_node_degrees, inputs,
                              max_workers=workers)

        today = ut.get_date_str()
        real_deg_path = os.path.join(out_dir, f'ACTUAL_DEGREES__date={today}.csv')
        approx_deg_path = os.path.join(out_dir, f'APPROX_DEGREES__date={today}.csv')

        real_dfs = []
        approx_dfs = []
        for out in outputs:
            rows = []
            for v in out['Node Degs']:
                rows.append([v, out['Node Degs'][v],
                             out['Node Hyper-Degs'][v]])
            df = pd.DataFrame(rows, columns=['Node', 'Degree', 'Hyper-degree'])
            for key in out:
                if key in ['Node Degs', 'Node Hyper-Degs']:
                    continue
                df[key] = out[key]
            if 'algo' not in out:
                real_dfs.append(df)
            else:
                approx_dfs.append(df)
        real_df_ = pd.concat(real_dfs)
        real_df_.to_csv(real_deg_path)
        approx_df_ = pd.concat(approx_dfs)
        approx_df_.to_csv(approx_deg_path)

    # REACHABILITY QUERY
    if query_exp[2]:
        # Generate queries
        node_samples = dict()
        for path in orig_paths:
            file_name = os.path.basename(path)
            dname = file_name.split('__')[1].split('=')[1].split('.')[0]
            nc, se, sw, _, _ = sut.load_summary(path, orig_node_id_maps[dname])
            nodes = list()
            for clust in nc:
                for v in clust:
                    nodes.append(v)
            samples = []
            for run in range(nruns):
                sample = ut.generate_random_pairs(nodes, sample_size)
                samples.append(sample)
            node_samples[dname] = samples
        # Answer queries
        inputs = []
        for run in range(nruns):
            for path in orig_paths:
                file_name = os.path.basename(path)
                dname = file_name.split('__')[1].split('=')[1].split('.')[0]
                inputs.append([path, False, node_samples[dname][run], run,
                               orig_node_id_maps[dname]])
            for path in summary_paths:
                file_name = os.path.basename(path)
                params = sut.get_params_from_summary_name(file_name)
                dataname = params['data'].split('.')[0]
                inputs.append([path, True, node_samples[dataname][run],
                               run, orig_node_id_maps[dataname]])
        outputs = process_map(parallel_reachability, inputs, max_workers=workers)

        today = ut.get_date_str()
        real_reach_path = os.path.join(out_dir, f'ACTUAL_REACHABLE__date={today}.csv')
        approx_reach_path = os.path.join(out_dir, f'APPROX_REACHABLE__date={today}.csv')

        real_dfs = []
        approx_dfs = []
        for out in outputs:
            rows = []
            for pair in out['Reachability']:
                rows.append([pair, out['Reachability'][pair]])
            df = pd.DataFrame(rows, columns=['Node Pair', 'Distance'])
            for key in out:
                if key in ['Reachability']:
                    continue
                df[key] = out[key]
            if 'algo' not in out:
                real_dfs.append(df)
            else:
                approx_dfs.append(df)
        real_df_ = pd.concat(real_dfs)
        real_df_.to_csv(real_reach_path)
        approx_df_ = pd.concat(approx_dfs)
        approx_df_.to_csv(approx_reach_path)

    # CONNECTED COMPONENTS QUERY
    if query_exp[3]:
        inputs = []
        for path in orig_paths:
            file_name = os.path.basename(path)
            dname = file_name.split('__')[1].split('=')[1].split('.')[0]
            inputs.append([path, False, orig_node_id_maps[dname]])
        for path in summary_paths:
            file_name = os.path.basename(path)
            params = sut.get_params_from_summary_name(file_name)
            dataname = params['data'].split('.')[0]
            inputs.append([path, True, orig_node_id_maps[dataname]])
        outputs = process_map(parallel_connected_components,
                              inputs, max_workers=workers)
        today = ut.get_date_str()
        real_ppr_path = os.path.join(out_dir, f'ACTUAL_CCS__date={today}.csv')
        approx_ppr_path = os.path.join(out_dir, f'APPROX_CCS__date={today}.csv')

        real_dfs = []
        approx_dfs = []
        for out in outputs:
            df = pd.DataFrame.from_dict(out, orient='index').T
            if 'algo' not in out:
                real_dfs.append(df)
            else:
                approx_dfs.append(df)
        real_df_ = pd.concat(real_dfs)
        real_df_.to_csv(real_ppr_path)
        approx_df_ = pd.concat(approx_dfs)
        approx_df_.to_csv(approx_ppr_path)

    # k-HOP NEIGHBORS QUERY
    if query_exp[4]:
        # Generate queries
        node_samples = dict()
        for path in orig_paths:
            file_name = os.path.basename(path)
            dname = file_name.split('__')[1].split('=')[1].split('.')[0]
            dname, sampl = ut.sample_node_set_from_hypergraph(path,
                                                              orig_node_id_maps[dname],
                                                              sample_size)
            node_samples[dname] = sampl
        # Answer queries
        inputs = []
        for h_d in [2, 3, 4]:
            for path in orig_paths:
                file_name = os.path.basename(path)
                dname = file_name.split('__')[1].split('=')[1].split('.')[0]
                inputs.append([path, False, node_samples[dname],
                               h_d, orig_node_id_maps[dname]])
            for path in summary_paths:
                file_name = os.path.basename(path)
                params = sut.get_params_from_summary_name(file_name)
                dataname = params['data'].split('.')[0]
                inputs.append([path, True, node_samples[dataname], h_d,
                               orig_node_id_maps[dataname]])
        outputs = process_map(parallel_reachable_at_k, inputs,
                              max_workers=workers)
        today = ut.get_date_str()
        real_reach_path = os.path.join(out_dir, f'ACTUAL_REACHABLE_AT_K__date={today}.csv')
        approx_reach_path = os.path.join(out_dir, f'APPROX_REACHABLE_AT_K__date={today}.csv')

        real_dfs = []
        approx_dfs = []
        for out in outputs:
            rows = []
            for node in out['Neighs at k']:
                node_ngb = out['Neighs at k'][node]
                for ngb in node_ngb:
                    rows.append([node, ngb])
            df = pd.DataFrame(rows, columns=['Node', 'Neigh at k'])
            for key in out:
                if key != 'Neighs at k':
                    df[key] = out[key]
            if 'algo' not in out:
                real_dfs.append(df)
            else:
                approx_dfs.append(df)
        real_df_ = pd.concat(real_dfs)
        real_df_.to_csv(real_reach_path)
        approx_df_ = pd.concat(approx_dfs)
        approx_df_.to_csv(approx_reach_path)

    # CLOSENESS CENTRALITY QUERY
    if query_exp[5]:
        # Generate queries
        node_samples = dict()
        for path in orig_paths:
            file_name = os.path.basename(path)
            dname = file_name.split('__')[1].split('=')[1].split('.')[0]
            dname, sampl = ut.sample_node_set_from_hypergraph(path, orig_node_id_maps[dname], sample_size)
            node_samples[dname] = sampl
        # Answer queries
        inputs = []
        for path in orig_paths:
            file_name = os.path.basename(path)
            dname = file_name.split('__')[1].split('=')[1].split('.')[0]
            inputs.append([path, False, node_samples[dname],
                           orig_node_id_maps[dname]])
        for path in summary_paths:
            file_name = os.path.basename(path)
            params = sut.get_params_from_summary_name(file_name)
            dataname = params['data'].split('.')[0]
            inputs.append([path, True, node_samples[dataname],
                           orig_node_id_maps[dataname]])
        outputs = process_map(parallel_closeness, inputs, max_workers=workers)

        today = ut.get_date_str()
        real_reach_path = os.path.join(out_dir, f'ACTUAL_CLOSENESS__date={today}.csv')
        approx_reach_path = os.path.join(out_dir, f'APPROX_CLOSENESS__date={today}.csv')

        real_dfs = []
        approx_dfs = []
        for out in outputs:
            rows = []
            for node in out['Closeness']:
                clos = out['Closeness'][node]
                rows.append([node, clos])
            df = pd.DataFrame(rows, columns=['Node', 'Closeness'])
            for key in out:
                if key != 'Closeness':
                    df[key] = out[key]
            if 'algo' not in out:
                real_dfs.append(df)
            else:
                approx_dfs.append(df)
        real_df_ = pd.concat(real_dfs)
        real_df_.to_csv(real_reach_path)
        approx_df_ = pd.concat(approx_dfs)
        approx_df_.to_csv(approx_reach_path)
