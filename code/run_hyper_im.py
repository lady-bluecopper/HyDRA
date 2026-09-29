import os
import pandas as pd
import random
import time
from typing import Dict, List, Tuple
from tqdm.contrib.concurrent import process_map

import sys
sys.path.append('.')
import utils as ut
import summary_utils as sut
from hn_moea_im.loaders import load_hypergraph
from hn_moea_im.smart_initialization import create_initial_population
from hn_moea_im.moea import moea_influence_maximization
from hn_moea_im.monte_carlo_max_hop import monte_carlo_max_hop_simulation
from hn_moea_im.upscaling import summary_upscaling_multi_sets


def parallel_IM(inp):
    data_path = inp[0]
    is_summary = inp[1]
    seed = inp[2]
    out_dir = inp[3]
    node_id_map = inp[4]
    args = inp[5]
    
    hyperG, node_clusters, _ = load_hypergraph(data_path, node_id_map)
    degree_dict:Dict[int,int] = dict()
    hyperdegree_dict:Dict[int,int] = dict()
    neighbor_dict:Dict[int,List[int]] = dict()
    incident_hyperedge_dict:Dict[int,List[Tuple[int]]] = dict()
    for n in hyperG.get_nodes():
        degree_dict[n] = len(hyperG.get_neighbors(n))
        hyperdegree_dict[n] = hyperG.degree(n)
        neighbor_dict[n] = hyperG.get_neighbors(n)
        incident_hyperedge_dict[n] = hyperG.get_incident_edges(n)

    rng = random.Random(seed)
    
    data_dir = os.path.dirname(data_path)
    fname = os.path.basename(data_path)
    activ_fp = os.path.join(out_dir, f'activation_attempts__seed={seed}__model={args["model"]}__input={fname}.csv')
    hypv_fp = os.path.join(out_dir, f'hypervolume__seed={seed}__model={args["model"]}__input={fname}.csv')
    
    actual_max = args['max_seed_nodes']
    args['max_seed_nodes'] = min(args['max_seed_nodes'], len(node_clusters))
    args['min_seed_nodes'] = min(args['min_seed_nodes'], len(node_clusters))

    start_time = time.time()
    # smart initialization
    initial_population = create_initial_population(
                                        hypergraph=hyperG,
                                        min_k=args['min_seed_nodes'],
                                        max_k=args['max_seed_nodes'],
                                        n=args['population_size'],
                                        prng=rng,
                                        strategy=args['init_strategy'])
    # run multi-objective evolutionary algorithm optimization
    pareto_front, final_pop = moea_influence_maximization(
                                        hypergraph=hyperG,
                                        degree_dict=degree_dict,
                                        hyperdegree_dict=hyperdegree_dict,
                                        neighbor_dict=neighbor_dict,
                                        incident_hyperedge_dict=incident_hyperedge_dict,
                                        random_gen=rng,
                                        initial_population=initial_population,
                                        custom_mutation=args['custom_mutation'],
                                        output_activation_attempts_file_path=activ_fp,
                                        output_hypervolume_file_path=hypv_fp,
                                        args=args)
    end_t = time.time() - start_time
    
    output = {
        'Time (s)': end_t,
        'Run': seed,
        'Seed Sets (pareto)': [index[0] for index in pareto_front], 
        'Node Perc. as Seed Set (pareto)': [index[2] for index in pareto_front],
        'Perc. Influenced Nodes (pareto)': [index[1] for index in pareto_front],
        'Seed Sets (last gen)': [index[0] for index in final_pop],
        'Node Perc. as Seed Set (last gen)': [index[2] for index in final_pop],
        'Perc. Influenced Nodes (last gen)': [index[1] for index in final_pop]
    }
    
    if is_summary:
        output_upscaling = []
        # read original hypergraph to test upscaled seed set
        summ_params = sut.get_params_from_summary_name(fname)
        orig_path = os.path.join(data_dir, f'original__data={summ_params['data']}.csv')
        orig_hg, _, _ = load_hypergraph(orig_path, node_id_map)
        orig_deg_dict = dict()
        orig_ngb_dict = dict()
        orig_inc_hedge_dict = dict()
        for n in orig_hg.get_nodes():
            orig_deg_dict[n] = len(orig_hg.get_neighbors(n))
            orig_ngb_dict[n] = orig_hg.get_neighbors(n)
            orig_inc_hedge_dict[n] = orig_hg.get_incident_edges(n)
        num_nodes = len(orig_hg.get_nodes())
        actual_max = min(actual_max, num_nodes)

        upscaled_seed_sets = summary_upscaling_multi_sets(
            output['Seed Sets (pareto)'],
            node_clusters,
            rng,
            max_seed_size=actual_max
        )
        for upscaled_seed_set in upscaled_seed_sets:
            mean_influenced, std_dev_influenced, _ = monte_carlo_max_hop_simulation(
                hypergraph=orig_hg,
                degree_dict=orig_deg_dict,
                neighbor_dict=orig_ngb_dict,
                incident_hyperedge_dict=orig_inc_hedge_dict,
                a=set(upscaled_seed_set),
                t=args['threshold'],
                p_min=args['p_min'],
                p_max=args['p_max'],
                no_simulations=args['no_simulations'],
                max_hop=args['max_hop'],
                model=args['model'],
                random_generator=rng)
            output_upscaling.append((mean_influenced / num_nodes, std_dev_influenced / num_nodes))
        output.update(summ_params)
        output['Seed Sets (upscaled)'] = upscaled_seed_sets
        output['Perc. Influenced Nodes (upscaled)'] = output_upscaling
    else:
        dname = fname.split('__')[1].split('=')[1]
        output['data'] = dname
    return output


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

    r = cfg['r']
    b = cfg['b']    
    min_size = cfg['min_size']
    max_trials = cfg['max_trials']
    max_no_improvements = cfg['max_no_improvements']
    custom_mutation = True if cfg['custom_mutation'] == 'True' else False
    
    args_im ={
        'min_seed_nodes': cfg['min_seed_nodes'],  # min size of a seed set
        'max_seed_nodes': cfg['max_seed_nodes'],  # max size of a seed set
        'population_size': cfg['population_size'],  # number of seed sets to create
        'offspring_size': cfg['offspring_size'],  # offspring of the EA
        'max_generations': cfg['max_generations'],  # maximum generations
        'tournament_size': cfg['tournament_size'],  # EA tournament size
        'mutation_rate': cfg['mutation_rate'],  # the rate at which mutation is performed
        'crossover_rate': cfg['crossover_rate'],  # the rate at which crossover is performed
        'num_elites': cfg['num_elites'],  # number of elites to consider
        'threshold': cfg['threshold'],  # threshold for LT propagation model
        'p_min': cfg['p_min'],  # probability MIN for SICP propagation model
        'p_max': cfg['p_max'],  # probability MAX for SICP propagation model
        'max_hop': cfg['max_hop'],  # maximum number of influence propagation time steps for SICP propagation model
        'model': cfg['model'],  # type of influence propagation model
        'no_simulations': cfg['no_simulations'],  # number of simulations for spread calculation
        'custom_mutation': custom_mutation,
        'n_threads': workers,
        'init_strategy': cfg['init_strategy']
    }
    
    # OUTPUT    
    os.makedirs(out_dir, exist_ok=True)
    summary_dir = os.path.join(out_dir, 'summaries')
    
    # ORIGINAL
    inputs = []
    for dataset in dataset_names:
        data_path = os.path.join(summary_dir, f'original__data={dataset}.csv')
        # load original hypergraph to get the node id map to use for loading the summaries
        _, _, node_id_map = load_hypergraph(data_path, dict())
        
        for seed in range(nruns):
            inputs.append([data_path, False, seed, out_dir, node_id_map, args_im])

            for algo in algo_names:
                for algseed in range(nruns):
                    sum_name = ut.get_summary_file_name(algo, dataset, algseed, cfg)
                    sum_path = os.path.join(summary_dir, sum_name)
                    inputs.append([sum_path, True, seed, out_dir, node_id_map, args_im])
    outputs = process_map(parallel_IM, inputs, max_workers=workers)
    
    today = ut.get_date_str()
    real_im_path = os.path.join(out_dir, f'ACTUAL_IM__model={args_im["model"]}__date={today}.csv')
    approx_im_path = os.path.join(out_dir, f'APPROX_IM__model={args_im["model"]}__date={today}.csv')
    approx_up_im_path = os.path.join(out_dir, f'APPROX_IM_UPSCALED__model={args_im["model"]}__date={today}.csv')
    
    real_dfs = []
    approx_dfs = []
    approx_up_dfs = []
    
    for out in outputs:
        keys = set(out.keys()).difference(['Seed Sets (pareto)',
                                           'Node Perc. as Seed Set (pareto)',
                                           'Perc. Influenced Nodes (pareto)',
                                           'Seed Sets (last gen)',
                                           'Node Perc. as Seed Set (last gen)',
                                           'Perc. Influenced Nodes (last gen)',
                                           'Seed Sets (upscaled)',
                                           'Perc. Influenced Nodes (upscaled)'])
        rows = []
        for idx, ss in enumerate(out['Seed Sets (pareto)']):
            rows.append([ss,
                         out['Node Perc. as Seed Set (pareto)'][idx],
                         out['Perc. Influenced Nodes (pareto)'][idx],
                         'Pareto'])
        for idx, ss in enumerate(out['Seed Sets (last gen)']):
            rows.append([ss,
                         out['Node Perc. as Seed Set (last gen)'][idx],
                         out['Perc. Influenced Nodes (last gen)'][idx],
                         'Final Population'])
        df = pd.DataFrame(rows, columns=['Seed Set', 
                                         '% Nodes in Seed Set',
                                         'Mean % Influenced Nodes',
                                         'Solution Type'])
        for key in keys:
            df[key] = out[key]
        if 'algo' not in out:
            real_dfs.append(df)
        else:
            approx_dfs.append(df)

        if 'Seed Sets (upscaled)' in out:
            rows_up = []
            for idx, ss in enumerate(out['Seed Sets (upscaled)']):
                rows_up.append([ss,
                                out['Perc. Influenced Nodes (upscaled)'][idx][0],
                                out['Perc. Influenced Nodes (upscaled)'][idx][1]])
            df = pd.DataFrame(rows_up, columns=['Seed Set',
                                                'Mean % Influenced Nodes',
                                                'STD % Influenced Nodes'])
            for key in keys:
                df[key] = out[key]
            approx_up_dfs.append(df)

    real_df_ = pd.concat(real_dfs)
    real_df_.to_csv(real_im_path)
    approx_df_ = pd.concat(approx_dfs)
    approx_df_.to_csv(approx_im_path)
    approx_up_df_ = pd.concat(approx_up_dfs)
    approx_up_df_.to_csv(approx_up_im_path)
