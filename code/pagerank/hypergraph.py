import numpy as np
from collections import defaultdict


class Hypergraph:
    def __init__(self):
        self.edges = []
        self.incident_edges = defaultdict(list)
        self.weights = []
        self.ID = {}
        self.ID_rev = {}

    def degree(self, v):
        return len(self.incident_edges[v])

    def w_degree(self, v):
        return sum(self.weights[e] for e in self.incident_edges[v])

    @property
    def n(self):
        # number of nodes
        return len(self.incident_edges)

    @property
    def m(self):
        #  number of hyperedges
        return len(self.edges)

    def add_edge(self, edge, w=1):
        eid = len(self.edges)
        self.edges.append(edge)
        for v in edge:
            self.incident_edges[v].append(eid)
        self.weights.append(w)
        self.ID[tuple(edge)] = eid
        self.ID_rev[eid] = edge

    def T(self, vec, v_init, alpha):
        res = np.zeros(self.n)
        eps = 1e-8

        for edge in self.edges:
            argmaxs, argmins = [], []
            maxval, minval = float('-inf'), float('inf')

            for v in edge:
                val = vec[v] / self.w_degree(v)
                if val > maxval + eps:
                    maxval, argmaxs = val, [v]
                elif val > maxval - eps:
                    argmaxs.append(v)

                if val < minval - eps:
                    minval, argmins = val, [v]
                elif val < minval + eps:
                    argmins.append(v)

            for v in argmaxs:
                res[v] += self.weights[self.ID[tuple(edge)]] * (maxval - minval) / len(argmaxs)
            for v in argmins:
                res[v] -= self.weights[self.ID[tuple(edge)]] * (maxval - minval) / len(argmins)

        res_init = vec.copy()
        res_init[v_init] -= 1
        return (1 - alpha) * res + alpha * res_init

    def T_round(self, vec, v_init, alpha, active_edges):
        res = np.zeros(self.n)
        eps = 1e-8

        for edge in active_edges:
            argmaxs, argmins = [], []
            maxval, minval = float('-inf'), float('inf')

            for v in edge:
                val = vec[v] / self.w_degree(v)
                if val > maxval + eps:
                    maxval, argmaxs = val, [v]
                elif val > maxval - eps:
                    argmaxs.append(v)

                if val < minval - eps:
                    minval, argmins = val, [v]
                elif val < minval + eps:
                    argmins.append(v)

            for v in argmaxs:
                res[v] += self.weights[self.ID[tuple(edge)]] * (maxval - minval) / len(argmaxs)
            for v in argmins:
                res[v] -= self.weights[self.ID[tuple(edge)]] * (maxval - minval) / len(argmins)

        res_init = vec.copy()
        res_init[v_init] -= 1
        return (1 - alpha) * res + alpha * res_init

    @staticmethod
    def from_list_to_HG(num_nodes: int,
                        hedges: list[list[int]], 
                        weights: list[int]):
        H = Hypergraph()
        for i in range(num_nodes):
            H.incident_edges[i] = list()
        H.edges = hedges
        H.weights = weights
        for idx, hedge in enumerate(hedges):
            for v in hedge:
                H.incident_edges[v].append(idx)
            H.ID[tuple(hedge)] = idx
            H.ID_rev[idx] = hedge
        return H

    @staticmethod
    def iterate(H, vec, v_init, dt, alpha):
        dv = H.T(vec, v_init, alpha)
        return vec - dv * dt

    @staticmethod
    def simulate(H, vec, v_init, dt, T, alpha):
        cur_time = 0.0
        while cur_time < T:
            next_time = min(cur_time + dt, T)
            vec = Hypergraph.iterate(H, vec, v_init,
                                     next_time - cur_time, alpha)
            cur_time = next_time
        return vec

    @staticmethod
    def iterate_round(H, vec, v_init, dt, alpha, active_edges, active):
        dv = H.T_round(vec, v_init, alpha, active_edges)
        res = vec - dv * dt

        res[res < 1e-5] = 0  # sparsification of PPR vector

        new_active_edges = []

        for i in range(H.n):
            if vec[i] == 0 and res[i] != 0:
                for f in H.incident_edges[i]:
                    if active[f] == 0:
                        new_active_edges.append(f)
                        active[f] = 1

        for e in new_active_edges:
            active_edges.append(H.ID_rev[e])

        return res

    @staticmethod
    def simulate_round(H, vec, v_init, dt, T, alpha):
        active_edges = []
        active = [0] * H.m

        for e in H.incident_edges[v_init]:
            active_edges.append(H.ID_rev[e])
            active[e] = 1

        cur_time = 0.0
        while cur_time < T:
            next_time = min(cur_time + dt, T)
            vec = Hypergraph.iterate_round(H, vec, v_init,
                                           next_time - cur_time,
                                           alpha, active_edges, active)
            cur_time = next_time
        return vec
