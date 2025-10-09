from typing import Dict, Tuple, List
import hypergraphx as hgx
import inspyred # type: ignore
import random

from hn_moea_im.monte_carlo_max_hop import monte_carlo_max_hop_simulation
from hn_moea_im.ea.observer import hypervolume_observer
from hn_moea_im.ea.terminator import generation_termination
from hn_moea_im.ea.generator import ea_generator
from hn_moea_im.ea.evaluator import ea_evaluator
from hn_moea_im.ea.crossover import ea_crossover
from hn_moea_im.ea.mutation import ea_mutation, ea_global_random_mutation
from hn_moea_im.ea.archiver import ea_archiver


def moea_influence_maximization(hypergraph: hgx.Hypergraph,
                                degree_dict:Dict[int,int],
                                hyperdegree_dict:Dict[int,int],
                                neighbor_dict:Dict[int,List[int]],
                                incident_hyperedge_dict:Dict[int,List[Tuple[int]]],
                                random_gen: random.Random,
                                initial_population: List[List[int]],
                                custom_mutation : bool,
                                output_activation_attempts_file_path : str,
                                output_hypervolume_file_path : str,
                                args: Dict):
    """
    Multi-objective evolutionary influence maximization.
        self.selector = selectors.default_selection
        self.variator = variators.default_variation
        self.migrator = migrators.default_migration
        self.population = None
        self.num_evaluations = 0
        self.num_generations = 0
    """
    # initialize multi-objective evolutionary algorithm NSGA-II
    fitness_function = monte_carlo_max_hop_simulation              # the influence is propagated up to a maximum number of hops

    ea = inspyred.ec.emo.NSGA2(random_gen)
    ea.archiver = ea_archiver                                      # archiver with Pareto preference (Pareto archive)
    if custom_mutation:
        ea.variator = [ea_crossover, ea_mutation]                  # type: ignore # the list of variation operators
    else:
        ea.variator = [ea_crossover, ea_global_random_mutation]    # type: ignore # the list of variation operators
    ea.observer = hypervolume_observer                             # the observer
    ea.terminator = generation_termination                         # the terminator

    hyperG_nodes = hypergraph.get_nodes()
    num_nodes = len(hyperG_nodes)
    
    # start the evolutionary process
    final_pop = ea.evolve(
        generator=ea_generator,                                     # the function to be used to generate candidate solutions 
        evaluator=ea_evaluator,                                     # the function to be used to evaluate candidate solutions
        bounder=inspyred.ec.DiscreteBounder(hyperG_nodes),          # a function used to bound candidate solutions
        maximize=True,                                              # boolean value stating use of maximization
        hypergraph=hypergraph,                                      # input hypergraph network
        nodes = hyperG_nodes,                                       # hypergraph nodes
        degree_dict=degree_dict,                                    # degree_dict[i] = degree node i
        hyperdegree_dict=hyperdegree_dict,                          # hyperdegree_dict[i] = hyperdegree node i
        neighbor_dict=neighbor_dict,                                # neighbor_dict[i] = list of neighbors of node i
        incident_hyperedge_dict=incident_hyperedge_dict,            # incident_hyperedge_dict[i] = list of incident hyperedges of node i
        random_generator=random_gen,                                # already initialized pseudo-random number generation
        seeds=initial_population,                                   # individuals (seed sets) to be added to the initial population (the rest will be randomly generated) 
        pop_size=args['population_size'],                           # the number of Individuals in the population 
        num_selected=args['offspring_size'],                        # offspring of the EA
        generations_budget=args['max_generations'],                 # maximum generations
        tournament_size=args['tournament_size'],                    # EA tournament size
        mutation_rate=args['mutation_rate'],                        # the rate at which mutation is performed
        crossover_rate=args['crossover_rate'],                      # the rate at which crossover is performed
        num_elites=args['num_elites'],                              # number of elites to consider
        p_min=args['p_min'],                                        # probability MIN for SICP propagation model
        p_max=args['p_max'],                                        # probability MAX for SICP propagation model
        threshold=args['threshold'],                                # threshold for LT propagation model
        max_hop=args['max_hop'],                                    # maximum number of influence propagation time steps for SICP propagation model
        propagation_model=args['model'],                            # type of influence propagation model
        no_simulations=args['no_simulations'],                      # number of simulations for spread calculation
        min_seed_nodes=args['min_seed_nodes'],                      # minimum number of nodes in a seed set
        max_seed_nodes=args['max_seed_nodes'],                      # maximum number of nodes in a seed set
        fitness_function=fitness_function,                          # fitness_function
        time=[],                                                    # keep track of Time (Activation Attempts) trend throughout the generations
        hypervolume=[],                                             # keep track of HV trend throughout the generations
        n_threads=args['n_threads'],                                # number of threads to handle parallel computation
        activation_attempts_file_path = output_activation_attempts_file_path,   # file path where to store the number of activation attempts
        hypervolume_file_path = output_hypervolume_file_path                    # file path where to store the hypervolume of the final population
    )

    # ea.archive contains the pareto fron of the population of candidate solutions evolved during the evolutionary process
    # individual.fitness[0] is the ratio of active nodes at the end of the process
    # individual.candidate is the seed set
    # final_pop includes the whole population of candidate solutions
    pareto_front = [[individual.candidate, individual.fitness[0] * 100,
                     (len(individual.candidate)  / num_nodes) * 100] for individual in ea.archive]  # type: ignore
    final_pop = [[individual.candidate, individual.fitness[0] * 100, 
                  (len(individual.candidate)  / num_nodes) * 100] for individual in final_pop]
    return pareto_front, final_pop
