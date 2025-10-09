import numpy as np
import time
import sys
sys.path.append('.')
from pagerank.hypergraph import Hypergraph


def integrate(f, T, dt=0.01):
    dummy = f(0.0)
    res = np.zeros(len(dummy))
    t = 0.0
    while t < T:
        nt = min(T - t, dt)
        res += f(t) * nt
        t += nt
    return res


def ppr(num_nodes: int,
        hedges: list[list[int]],
        weights: list[int],
        v_init: int = -1,  # id of the starting node
        alpha: float = 0.85,
        seed: int = 42,
        T: int=30, # total time
        verbose=False):

    np.random.seed(seed)

    H = Hypergraph.from_list_to_HG(num_nodes, hedges, weights)
    n = H.n
    dt = 1.0  # time span

    if v_init < 0:
        v_init = np.random.randint(n)

    start_time = time.time()
    vec = np.zeros(n)
    vec[v_init] = 1.0
    vec = Hypergraph.simulate_round(H, vec, v_init, dt, T, alpha)  # PPR
    if verbose:
        print("Time (s):", time.time() - start_time)
    return vec, time.time() - start_time
