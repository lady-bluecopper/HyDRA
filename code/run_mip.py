import numpy as np
from collections import defaultdict
from pyomo.environ import *


if __name__ == '__main__':

    hedges = [
        [1,2,3,4,5,6],
        [1,2,4,5],
        [2,3,4,5],
        [1,2,3,4,5],
        [1,4,5,7,0],
        [1,2,3,6,7,0,10,11],
        [1,2,3,6,7,10,11],
        [2,3,6,7,10,11],
        [1,2,6,7,8,10,11],
        [2,3,6,7,9,10,11],
        [1,2,6,7,10,11],
        [6,7,8,9,0],
        [4,6,7,8,9],
        [4,5,6,7,9,0],
        [6,8,9,0],
        [8,9,0],
        [2,8,9,0],
        [3,7,8,9],
        [6,7,0]
    ]

    vertices = set()
    for h in hedges:
        vertices.update(h)

    n = len(vertices)
    m = len(hedges)

    A = np.zeros((n,m))
    for hid, hedge in enumerate(hedges):
        for v in hedge:
            A[v, hid] = 1

    # Model
    model = ConcreteModel()
    # Variables
    # Node - Supernode
    model.X = Var(range(n), range(n), domain=Binary)
    # Supernode - Superedge
    model.Y = Var(range(n), range(m), domain=Binary)
    # Hyperedge - Superedge
    model.Z = Var(range(m), range(m), domain=Binary)
    # Node - Hyperedge
    model.V = Var(range(n), range(m), domain=Binary)
    # Superedge size
    model.G = Var(range(m), domain=Binary)

    # Constraints
    model.constraints = ConstraintList()

    # Coherence Constraints
    for i in range(n):
        for j in range(n):
            for l in range(m):
                for k in range(m):
                    model.constraints.add((1 - model.X[i, j]) + (1 - model.Y[j, l]) + (1 - model.Z[k, l]) + model.V[i, k] >= 1)
                    model.constraints.add((1 - model.X[i, j]) + (1 - model.Z[k, l]) + (1 - model.V[i, k]) + model.Y[j, l] >= 1)
    # Each node must belong to only one cluster
    for i in range(n):
        model.constraints.add(sum(model.X[i, j] for j in range(n)) == 1)
    # Each edge must belong to only one cluster
    for i in range(m):
        model.constraints.add(sum(model.Z[i, j] for j in range(m)) == 1)
    # Non-empty superedges
    for i in range(m):
        for j in range(m):
            model.constraints.add(model.G[i] >= model.Z[j,i])

    def objective_rule(model):

        # Size of each super-hyperedge
        term1 = sum(sum(model.Y[i, j] for j in range(m)) for i in range(n))
        # Number of corrections
        term2 = sum(
            A[i, j] * (1 - model.V[i, j]) + (1 - A[i, j]) * model.V[i, j]
            for i in range(n) for j in range(m)
        )
        # Number of super-hyperedges
        term3 = sum(model.G[i] for i in range(m))

        return term1 + term2 + term3

    model.obj = Objective(rule=objective_rule, sense=minimize)

    solver = SolverFactory('gurobi_direct', manage_env=True)
    solver.options['NodefileStart'] = .5
    solver.options['Threads'] = 30
    solver.options['Presolve'] = 2
    solver.options['MIPGap'] = 1e-1
    solver.options['Cuts'] = 2
    solver.options['MIPFocus'] = 3 # 2 means focus on proving optimality; 3 means focus on bound
    try:
        results = solver.solve(model, tee=True)
        print(model.obj())
        print(results.solver.status)
        print(results.solver.termination_condition)
    finally:
        solver.close()

    hedge_cluster_id = dict()
    for h in range(m):
        for h2 in range(m):
            if model.Z[h,h2].value > 0.5:
                if h in hedge_cluster_id:
                    print(f'WARN: {h} already present with id {hedge_cluster_id[h]}')
                hedge_cluster_id[h] = h2
    hedge_clusters = defaultdict(list)
    for h, cid in hedge_cluster_id.items():
        hedge_clusters[cid].append(h)

    node_cluster_id = dict()
    for v in range(n):
        for v2 in range(n):
            if model.X[v,v2].value > 0.5:
                if v in node_cluster_id:
                    print(f'WARN: {v} already present with id {node_cluster_id[v]}')
                node_cluster_id[v] = v2
    node_clusters = defaultdict(list)
    for h, cid in node_cluster_id.items():
        node_clusters[cid].append(h)

    superH = defaultdict(set)
    for supN in node_clusters.keys():
        for supE in hedge_clusters.keys():
            if model.Y[supN,supE].value > 0.5:
                superH[supE].add(supN)

    corr_table = dict()
    for cid, hedge_cluster in hedge_clusters.items():
        this_corrections = []
        for hedge in hedge_cluster:
            Nh = set(hedges[hedge])
            Nc = set()
            for supN in superH[cid]:
                Nc.update(node_clusters[supN])
            cPlus = Nh.difference(Nc)
            cMinus = Nc.difference(Nh)
            this_corrections.append((cPlus, cMinus))
        corr_table[cid] = this_corrections

    # cost of storing the weights
    e1 = len(superH)
    # cost of storing the superhyperedges
    e2 = sum([len(se) for se in superH.values()])
    # cost of corrections
    e3 = 0
    for corr_lst in corr_table.values():
        for corr in corr_lst:
            e3 += len(corr[0])
            e3 += len(corr[1])
    cost = e1 + e2 + e3

    print('cost', cost)
    print('node clusters')
    print(node_clusters)
    print('superedges')
    print(superH)
