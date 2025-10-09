from typing import List
import hypergraphx as hgx
import random


def create_initial_population(hypergraph: hgx.Hypergraph,
                              min_k: int,
                              max_k: int,
                              n: int,
                              prng: random.Random,
                              strategy: str='smart') -> List[List[int]]:
    """
    Parameters
    ----------
    hypergraph : hgx.Hypergraph
        Hypergraphx Hypergraph object which encodes the input network.
    min_k : int
        Minimum size of the seed set of the individuals belonging to the initial population.
    max_k : int
        Maximum size of the seed set of the individuals belonging to the initial population.
    n : int
        Number of individuals of the initial population.
    degree_function : Calleble[[hgx.Hypergraph, int], int]
        Degree or hyperdegree of input node n
    prng : random.Random
        Pseudo-random generator.

    Returns
    -------
        Initial population.
    """

    if strategy == 'smart':
        return smart_initialization(hypergraph,
                                    min_k,
                                    max_k,
                                    n,
                                    prng)
    elif strategy == 'degree':
        return high_degree_discount_initialization(hypergraph, max_k)
    raise NotImplementedError


def smart_initialization(hypergraph: hgx.Hypergraph,
                         min_k: int,
                         max_k: int,
                         n: int,
                         prng: random.Random) -> List[List[int]]:
    """
    Apply smart initialization of the initial population.
    - Apply node filtering, which considers only the nodes with the highest
    metric as defined by parameter degree_function.
    - Each of these nodes is added to a candidate solution with a probability
    proportional to its degree.

    Parameters
    ----------
    hypergraph : hgx.Hypergraph
        Hypergraphx Hypergraph object which encodes the input network.
    min_k : int
        Minimum size of the seed set of the individuals belonging to the initial population.
    max_k : int
        Maximum size of the seed set of the individuals belonging to the initial population.
    n : int
        Number of individuals of the initial population.
    degree_function : Calleble[[hgx.Hypergraph, int], int]
        Degree or hyperdegree of input node n
    prng : random.Random
        Pseudo-random generator.

    Returns
    -------
        Initial population.
    """
    individuals = []
    # half of the initial population comprises seed sets of nodes
    # chosen uniformly at random from the entire node set V
    for _ in range(int(n // 2)):
        # extract random number in 1,max_seed_nodes and initialize individual genome
        individual_size = random.randint(min_k, max_k)
        individuals.append(prng.sample(hypergraph.get_nodes(), individual_size))

    # select a subset of nodes characterized by high degree centrality
    all_nodes = hypergraph.get_nodes()
    all_nodes_degree = [len(hypergraph.get_neighbors(node)) for node in all_nodes] # type: ignore
    if sum(all_nodes_degree) == 0:
        # this can happen for summaries because nodes can be disconnected
        all_nodes_degree = None
    num_nodes_filtered = max_k + prng.randint(0, (len(hypergraph.get_nodes()) - max_k) // 2)
    nodes_filtered = prng.choices(all_nodes, weights=all_nodes_degree, k=num_nodes_filtered)

    # choose n/2 individuals containing k nodes, chosen from the input hypergraph
    # with probabilities proportional to their degrees.
    sorted_nodes = sorted(nodes_filtered)
    nodes_degree = [len(hypergraph.get_neighbors(node)) for node in sorted_nodes]
    for _ in range(n//2):
        # copy because we will modify it
        nodes_ = sorted_nodes.copy() # type: ignore
        probs_ = nodes_degree.copy()
        if sum(probs_) == 0:
            probs_ = [1 for node in sorted_nodes]
        new_individual = []
        new_individual_size = prng.randint(min_k, max_k)

        for _ in range(new_individual_size):
            new_node = prng.choices(nodes_, probs_)[0]
            probs_.pop(nodes_.index(new_node))
            # so that it will not be selected again
            nodes_.remove(new_node)
            new_individual.append(new_node)
        individuals.append(new_individual)
    return individuals


def high_degree_discount_initialization(hypergraph: hgx.Hypergraph,
                                        k: int) -> List[List[int]]:
    """
    Execute High Degree Discount algoritm as proposed in:
    https://arxiv.org/abs/2206.01394

    Parameters
    ----------
    hypergraph : hgx.Hypergraph
        Hypergraphx Hypergraph object which encodes the input network.
    
    k : int
        Cardinality of the seed set.

    Returns
    -------
        Seed set of k nodes selected by HDD optimization algorithm.
    """
    seeds = set()

    degree = {}
    for n in hypergraph.get_nodes():
        degree[n] = len(hypergraph.get_neighbors(n)) # type: ignore

    for _ in range(k):
        # sort nodes according to their adaptive degree
        sorted_nodes = sorted(degree.keys(), key=lambda x: degree[x], reverse=True)

        # select the node with the largest adaptive degree which is not in the
        # seed set yet and add it to the seed set
        for node in sorted_nodes:
            if node not in seeds:
                chosenNode = node
                break
        seeds.add(chosenNode)
        # update adaptive degree
        for v_q in hypergraph.get_neighbors(chosenNode): # type: ignore
            z = 0
            for v_q_neighbor in hypergraph.get_neighbors(v_q):
                if v_q_neighbor in seeds:
                    z += 1
            degree[v_q] = degree[v_q] - z

    return [list(seeds)]