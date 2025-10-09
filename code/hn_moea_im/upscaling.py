from typing import List, Set
from tqdm import tqdm
import random
import numpy as np


def summary_upscaling(summary_seed_set: List[List[int]], 
                      supernodes: List[List[int]], 
                      rng: random.Random):
    """
    Given the seed set obtained executing the evolutionary process on the summary
    network, return the corresponding seed set of nodes of the initial hypergraph.
    In order to shift from seed set of supernodes to seed set of nodes we sample
    a component uniformly at random from each supernode which belongs to the set.

    Parameters
    ----------
    summary_seed_set : list[list]
        This input parameter represents the final population obtained at the end
        of the evolutionary process executed on the hypergraph summary network.
        Example: [[[602], 90.0, 2.0], [[550, 598], 96.0, 4.0], [[553], 90.0, 2.0]]
    
    supernodes : dict[int, set[int]]
        For each supernode id (key), the set of components of the original
        hypergraph which populate the supernode.
    
    random_generator : random.Random
        Already initialized random number generator.

    Returns
    -------
    The upscaled seed set of components.
    """
    # upscale the pareto front of the summary to the corresponding original hypergraph components
    upscaled_pareto = [[rng.choice(supernodes[supN]) for supN in seed_set] for seed_set in summary_seed_set]
    return upscaled_pareto


def summary_upscaling_multi_sets(summary_seed_set: List[List[int]], 
                                 supernodes: List[List[int]], 
                                 rng: random.Random,
                                 max_seed_size: int,
                                 max_samples: int=5) -> List[Set[int]]:
    """
    Given the seed set obtained executing the evolutionary process on the summary
    network, return the corresponding seed set of nodes of the initial hypergraph.
    In order to shift from seed set of supernodes to seed set of nodes we sample
    a component uniformly at random from each supernode which belongs to the set.

    Parameters
    ----------
    summary_seed_set : list[list]
        This input parameter represents the final population obtained at the end
        of the evolutionary process executed on the hypergraph summary network.
        Example: [[[602], 90.0, 2.0], [[550, 598], 96.0, 4.0], [[553], 90.0, 2.0]]
    
    supernodes : dict[int, set[int]]
        For each supernode id (key), the set of components of the original
        hypergraph which populate the supernode.
    
    random_generator : random.Random
        Already initialized random number generator.
    Returns
    -------
    The upscaled seed set of components.
    """
    supN_sizes = np.array([len(supN) for supN in supernodes])
    upscaled_sets = []

    for seed_set in tqdm(summary_seed_set):
        set_num_supN = len(seed_set)
        num_avail_nodes = sum(x for x in supN_sizes[seed_set])
        set_max_seed_size = min(max_seed_size, num_avail_nodes)
        step = max(1, (set_max_seed_size - set_num_supN) // max_samples)
        
        for seed_size in range(set_num_supN, set_max_seed_size, step):
            set_supN_list = [set(supernodes[x]) for x in seed_set]
            upscaled_set = set()
            if set_num_supN == 1:
                upscaled_set = set(rng.choices(population=supernodes[seed_set[0]], k=seed_size))
            elif seed_size == num_avail_nodes:
                for supN in seed_set:
                    for n in supernodes[supN]:
                        upscaled_set.add(n)
            else:
                for i in range(set_num_supN):
                    if len(set_supN_list[i]) > 0:
                        upscaled_set.add(set_supN_list[i].pop())
                res_weights = [len(x) for x in set_supN_list]
                while len(upscaled_set) < seed_size:
                    print('current size', len(upscaled_set), 'needed more', seed_size - len(upscaled_set), 'nodes', 'res_weights', res_weights, 'set_max_seed_size', set_max_seed_size)
                    supN_chosen = rng.choices(population=range(set_num_supN), weights=res_weights, k=1)[0]
                    new_node = set_supN_list[supN_chosen].pop()
                    upscaled_set.add(new_node)
                    res_weights[supN_chosen] -= 1
                    if sum(res_weights) == 0:
                        break
            upscaled_sets.append(upscaled_set)
    return upscaled_sets
