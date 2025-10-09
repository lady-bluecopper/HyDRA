from helpers import compute_Dnz_opt, compute_Dnz2_opt, generate_candidate_groups_jac, logstar2, generate_candidate_groups, entropy_bits, end_time_ms
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import time
import sys
sys.path.append('.')


def init_perm(r: int, b: int, A: csr_matrix):
    m, n = A.shape
    Nx = np.ones(m)
    Ny = np.ones(n)
    Qx = np.arange(m)
    Qy = np.arange(n)
    Ny, Qy = jac_col_hash2(A, r, b, Ny, Qy)
    Nx, Qx = jac_col_hash2(A.T, r, b, Nx, Qx)
    k, l, Nx, Ny, DnZ = compute_Dnz_opt(Qx, Nx, Qy, Ny, A)
    return k, l, Nx, Ny, Qx, Qy, DnZ


def init_cost(A):
    k, l = A.shape
    # 1. Encoding cost for k and l
    c = logstar2(k) + logstar2(l)
    # 2. Encoding cost for row cluster sizes
    c += k * entropy_bits(1 / k)
    # 3. Encoding cost for column cluster sizes
    c += l * entropy_bits(1 / l)
    # 4. Encoding cost for number of 1s in each block log2(rici+1)=log2(1+1)=1
    c += k * l
    return c


def cc_cost(k, l, NX, NY, DnZ):
    """
    Parameters:
    k, l   : Number of row and column clusters.
    Nx, Ny : Arrays of row and column cluster sizes.
    Dnz    : 2D array containing the number of non-zeros in each block.

    Returns:
    c      : Total encoding cost.
    """

    Dnz = DnZ.copy()
    Nx = NX.copy()
    Ny = NY.copy()

    # Step 0: Disregard empty clusters
    real_k = np.count_nonzero(Nx)
    real_l = np.count_nonzero(Ny)
    if real_k != k or real_l != l:
        k = real_k
        l = real_l
        non_empty_row_clusters = np.nonzero(Nx)[0]
        non_empty_col_clusters = np.nonzero(Ny)[0]
        Nx = Nx[non_empty_row_clusters]
        Ny = Ny[non_empty_col_clusters]
        Dnz = Dnz[np.ix_(non_empty_row_clusters, non_empty_col_clusters)]

    # Step 1: Encoding cost for k and l
    c = logstar2(k) + logstar2(l)
    # Step 2: Encoding cost for row cluster sizes
    entrow = Nx * entropy_bits(Nx / np.sum(Nx))
    c += np.sum(entrow)
    # Step 3: Encoding cost for column cluster sizes
    entcol = Ny * entropy_bits(Ny / np.sum(Ny))
    c += np.sum(entcol)
    # Step 4.0a: Compute Nxy, the size of each cluster
    Nxy = np.outer(Nx, Ny)
    # Step 4.0b: Compute number of zeros
    Dz = csr_matrix(Nxy - Dnz)
    # Step 4.1: Encoding cost for the number of non-zeros
    for i in range(k):
        for j in range(l):
            c += np.log2(Nxy[i, j] + 1)
    # Step 4.1: Encoding cost for data in each block
    with np.errstate(divide='ignore', invalid='ignore'):
        Pz = (Dz / Nxy).toarray() # type: ignore
        Pz[~np.isfinite(Pz)] = 0  # Handle division by zero
        Pnz = (Dnz / Nxy).toarray()
        Pnz[~np.isfinite(Pnz)] = 0
    fst = Dz.multiply(csr_matrix(entropy_bits(Pz)))
    snd = Dnz.multiply(csr_matrix(entropy_bits(Pnz)))
    entropy_terms = fst + snd
    c2 = np.ceil(entropy_terms.sum())
    c += c2
    return c


def cos_col_hash2(r, b, k0, l0, Nx, NY, QY, DnZ, 
                  min_size: int=2,
                  max_trials: int=5):
    """
    Compute random-projection signatures and hash columns into buckets.

    Parameters:
        r (int): Number of rows in each hash signature block.
        b (int): Number of hash tables.
        k0 (int): Number of row clusters.
        l0 (int): Number of column clusters.
        Nx (numpy array): Row cluster sizes.
        Ny (numpy array): Column cluster sizes.
        Qy (numpy array): Column cluster assignments.
        DnZ (numpy array): Non-zero entries matrix.

    Returns:
        Ny (numpy array): Updated column cluster sizes.
        Qy (numpy array): Updated column cluster assignments.
        cluster_ids (numpy array): Updated cluster IDs.
    """
    Dnz = DnZ.copy()
    Ny = NY.copy()
    Qy = QY.copy()

    # Compute probabilities
    Nxy = np.outer(Nx, Ny)
    Dz = Nxy - Dnz

    with np.errstate(divide='ignore', invalid='ignore'):
        Pnz = (Dnz / Nxy).toarray()
        Pnz[~np.isfinite(Pnz)] = 0
        Pz = Dz / Nxy
        Pz[~np.isfinite(Pz)] = 0

    cluster_ids = np.unique(Qy)
    assert len(cluster_ids) == len(Ny)
    assert len(cluster_ids) == np.sum(Ny > 0)
    assert len(cluster_ids) == l0
    
    ucg, candidategroups = generate_candidate_groups(r, b, k0, l0, Pnz)

    m = len(Qy)
    # Step 4: Merge candidate groups
    for uc in ucg:
        ind = np.where(candidategroups == uc)[0]
        num = len(ind)
        if num < min_size:
            continue

        curl0 = l0
        trials = 0
        while True:
            merged = False
            ix = np.where(Ny[ind] > 0)[0]  # take only non-empty clusters
            if len(ix) == 0:
                break
            # pick random element from the group ind
            fo = np.random.choice(ix)
            cand1 = [ind[fo]]

            Dnzc = Dnz[:, cand1[0]].copy()
            Pnzc = Pnz[:, cand1[0]: cand1[0] + 1].copy()
            Dzc = Dz[:, cand1[0]].copy()
            Pzc = Pz[:, cand1[0]].copy()

            for o in range(num):  # for all the other elements
                if ind[o] in cand1 or Ny[ind[o]] == 0:
                    continue
                # ############# EQUATION 2
                curcost = 0
                #  reduce num cluster cost
                curcost -= logstar2(curl0 - len(cand1) + 1)
                curcost += logstar2(curl0 - len(cand1))
                # reduce cluster assignment cost
                cis = [np.sum(Ny[cand1]), Ny[ind[o]]]
                curcost -= np.sum(cis * np.log2(m / np.array(cis)))
                curcost += np.sum(cis) * np.log2(m / np.sum(cis))
                #  reduce encoding cost of 1s
                curcost -= np.sum(np.sum(np.log2(np.outer(Nx, cis) + 1)))
                curcost += np.sum(np.log2(Nx * sum(cis) + 1))
                assert curcost <= 0

                Dnzm = sp.hstack((Dnzc, Dnz[:, ind[o]].reshape(-1, 1)))
                Pnzm = np.hstack([Pnzc, Pnz[:, ind[o]:ind[o] + 1]])
                Dzm = np.hstack([Dzc, Dz[:, ind[o]:ind[o] + 1]])
                Pzm = np.hstack([Pzc, Pz[:, ind[o]:ind[o] + 1]])
                entropy_terms = np.multiply(Dzm, entropy_bits(Pzm)) +\
                    Dnzm.multiply(entropy_bits(Pnzm))
                oldc = entropy_terms.sum().sum()
                curcost -= oldc

                Dnzv = Dnzm.sum(axis=1)
                newrowsizes = np.dot(Nx, sum(cis)).reshape(-1, 1)
                Pnzv = Dnzv / newrowsizes
                Dzv = newrowsizes - Dnzv
                Pzv = 1 - Pnzv
                entropy_terms = np.multiply(Dzv, entropy_bits(Pzv)) +\
                    np.multiply(Dnzv, entropy_bits(Pnzv))
                newc = np.sum(entropy_terms)
                curcost += newc

                assert newc - oldc >= 0 or np.abs(oldc - newc) < 0.000001

                ###################
                if curcost <= 0:
                    merged = True
                    cand1.append(ind[o])

                    Dnzc = Dnzv
                    Pnzc = Pnzv
                    Dzc = Dzv
                    Pzc = Pzv

            if len(cand1) > 1:

                curl0 -= len(cand1) + 1

                newind = np.array(cand1)
                cis = Ny[newind]

                minid = np.min(cluster_ids[newind])
                minind = np.argmin(cluster_ids[newind])
                Ny[newind] = 0
                Ny[newind[minind]] = np.sum(cis)

                for idx in newind:
                    if cluster_ids[idx] != minid:
                        Qy[Qy == cluster_ids[idx]] = minid
                        cluster_ids[cluster_ids == cluster_ids[idx]] = minid
                Dnz[:, newind[minind]] = Dnzc
                # ---- TODO: modified
                # Dnz_lil = Dnz.tolil()
                # Dnz_lil[:, newind[minind]] = Dnzc
                # Dnz = Dnz_lil.tocsr()
                Pnz[:, newind[minind]:newind[minind] + 1] = Pnzc
                Dz[:, newind[minind]] = Dzc
                Pz[:, newind[minind]] = Pzc

            if not merged:
                trials += 1
            if r == 1:
                if trials > max_trials:
                    break
            elif trials > 0:
                break
    return Ny, Qy, cluster_ids


def jac_col_hash2(A, r, b, NY, QY, min_size=5):
    '''
    Ny: sizes of column clusters
    Qy: cluster ids of columns
    '''
    Ny = NY.copy()
    Qy = QY.copy()
    n, m = A.shape
    
    ucg, candidategroups = generate_candidate_groups_jac(r, b, A)
    # try merging candidate GROUPS, don't if cost larger
    for uc in ucg:
        ind = np.where(candidategroups == uc)[0]
        num = len(ind)
        if num < min_size:
            continue
        # try to merge not all at once but 1 by 1
        cand1 = [ind[0]]
        Dnzc = A[:, cand1[0]].copy()
        Pnzc = A[:, cand1[0]].copy()
        Dzc = csr_matrix(np.ones(Dnzc.shape)) - Dnzc
        Pzc = csr_matrix(np.ones(Pnzc.shape)) - Pnzc

        for o in range(1, num):
            # merge group-cand1 with group ind(o)?
            cis = np.array([np.sum(Ny[cand1]), 1])
            # reduce cluster assignment cost
            curcost = 0
            curcost -= np.sum(cis * np.log2(m / cis))
            curcost += np.sum(cis) * np.log2(m / np.sum(cis))
            # reduce encoding cost of 1s
            curcost -= n * (np.sum(np.log2(np.array(cis) + 1)))
            curcost += n * (np.log2(sum(cis) + 1))
            # increase block-encoding cost
            # subtract
            Dnzm = sp.hstack((Dnzc, A[:, ind[o]].reshape(-1, 1)))
            Pnzm = sp.hstack((Pnzc, A[:, ind[o]].reshape(-1, 1)))
            A1 = csr_matrix(np.ones(A[:, ind[o]].shape))
            Dzm = sp.hstack((Dzc, (A1 - A[:, ind[o]]).reshape(-1, 1)))
            Pzm = sp.hstack((Pzc, (A1 - A[:, ind[o]]).reshape(-1, 1)))
            entropy_terms = Dzm.multiply(entropy_bits(Pzm.toarray())) +\
                Dnzm.multiply(entropy_bits(Pnzm.toarray()))
            oldc = entropy_terms.sum()
            curcost -= oldc

            Dnzv = Dnzm.sum(axis=1)
            Pnzv = Dnzv / np.sum(cis)
            Dzv = np.sum(cis) - Dnzv
            Pzv = 1 - Pnzv
            entropy_terms = np.multiply(Dzv, entropy_bits(Pzv)) +\
                np.multiply(Dnzv, entropy_bits(Pnzv))
            newc = np.sum(entropy_terms)
            curcost += newc

            # check if better cost
            if curcost <= 0:
                cand1.append(ind[o])
                Dnzc = Dnzv
                Pnzc = Pnzv
                Dzc = Dzv
                Pzc = Pzv
        if len(cand1) > 1:
            newind = cand1
            # Ny and Qy are going to change
            Ny[newind] = 0
            Ny[min(newind)] = len(newind)
            Qy[newind] = min(newind)
    return Ny, Qy


def hc_search(A, 
              r=20, 
              b=5, 
              seed=42, 
              cost_threshold=1e-6,
              min_size: int=2,
              max_trials: int=5,
              max_no_improvements: int=2,
              verbose=False):
    """
    Parameters:
    A: numpy.ndarray
        Binary adjacency matrix (nxm).
    r, b: int
        LSH parameters (recommended r=20, b=5 for practitioners).
    OUTPUT
    ======
    k: number of row clusters
    l: number of col clusters
    Nx: row cluster sizes
    Ny: col cluster sizes
    Qx: row cluster assignments
    Qy: col cluster assignments
    DnZ: matrix of number of non-zeros in each block
    c0, float: final cost
    A: input adjacency matrix with shuffled rows and columns
    """
    # Initialize costs and variables
    np.random.seed(seed)
    cinit = init_cost(A)
    # Initialize structures to store results
    start_comp = time.time_ns()
    stats = []
    if verbose:
        print(f"Starting cost is {cinit:.6f} bits")

    k, l, Nx, Ny, Qx, Qy, DnZ = init_perm(r, b, A)
    end_comp = end_time_ms(start_comp)
    start_comp = time.time_ns()
    c0 = cc_cost(k, l, Nx, Ny, DnZ)
    stats.append(['JAC', end_comp, cinit - c0, 0])
    stats.append(['JAC-CC', end_time_ms(start_comp), cinit - c0, 0])
    if verbose:
        print(f"Cost after 1st iterations: {c0:.6f}")

    no_improvements = 0
    it = 1
    r1 = r
    b1 = b
    C = 1.2
    while True:
        start_comp = time.time_ns()
        Ny1, Qy1, c_cl_ids = cos_col_hash2(round(r1 / (C**(it - 1))),
                                           round(b1 * (C**(it - 1))),
                                           k, l, Nx, Ny, Qy, DnZ,
                                           min_size=min_size,
                                           max_trials=max_trials)
        end_comp = end_time_ms(start_comp)
        l1, Ny1, DnZ1 = compute_Dnz2_opt(Ny1, DnZ, c_cl_ids)
        start_comp = time.time_ns()
        c1 = cc_cost(k, l1, Nx, Ny1, DnZ1)
        stats.append(['COS-COL-rb', end_comp, c0 - c1, it])
        stats.append(['COS-COL-rb-CC', end_time_ms(start_comp), c0 - c1, it])
        if c0 - c1 < cost_threshold:
            no_improvements += 1
        else:
            no_improvements = 0
            c0 = c1
            l, Ny, DnZ, Qy = l1, Ny1, DnZ1, Qy1
            if verbose:
                print(f"Cost after col-cluster merge: {c0:.6f}")
        start_comp = time.time_ns()
        Nx1, Qx1, r_cl_ids = cos_col_hash2(round(r1 / (C**(it - 1))),
                                           round(b1 * (C**(it - 1))),
                                           l, k, Ny, Nx, Qx, DnZ.T,
                                           min_size=min_size,
                                           max_trials=max_trials)
        end_comp = end_time_ms(start_comp)
        k1, Nx1, DnZp = compute_Dnz2_opt(Nx1, DnZ.T, r_cl_ids)
        DnZ1 = DnZp.T
        start_comp = time.time_ns()
        c1 = cc_cost(k1, l, Nx1, Ny, DnZ1)
        stats.append(['COS-ROW-rb', end_comp, c0 - c1, it])
        stats.append(['COS-ROW-rb-CC', end_time_ms(start_comp), c0 - c1, it])
        if c0 - c1 < cost_threshold:
            no_improvements += 1
        else:
            no_improvements = 0
            c0 = c1
            k = k1
            Nx = Nx1
            DnZ = DnZ1
            Qx = Qx1
            if verbose:
                print(f"Cost after row-cluster merge: {c0:.6f}")
        if no_improvements >= max_no_improvements:
            break
        it += 1

    if verbose:
        print(f'Cost after first While cycle: {c0}\n')

    no_improvements = 0
    it = 1

    while True:
        start_comp = time.time_ns()
        Ny1, Qy1, c_cl_ids = cos_col_hash2(1, 100, k, l, Nx, Ny, Qy, DnZ,
                                           min_size=min_size,
                                           max_trials=max_trials)
        end_comp = end_time_ms(start_comp)
        l1, Ny1, DnZ1 = compute_Dnz2_opt(Ny1, DnZ, c_cl_ids)
        start_comp = time.time_ns()
        c1 = cc_cost(k, l1, Nx, Ny1, DnZ1)
        stats.append(['COS-COL', end_comp, c0 - c1, it])
        stats.append(['COS-COL-CC', end_time_ms(start_comp), c0 - c1, it])
        if c0 - c1 < cost_threshold:
            no_improvements += 1
        else:
            no_improvements = 0
            c0 = c1
            l = l1
            Ny = Ny1
            DnZ = DnZ1
            Qy = Qy1
            if verbose:
                print(f'Cost after col-cluster merge: {c0}\n')

        start_comp = time.time_ns()
        Nx1, Qx1, r_cl_ids = cos_col_hash2(1, 100, l, k, Ny, Nx, Qx, DnZ.T,
                                           min_size=min_size,
                                           max_trials=max_trials)
        end_comp = end_time_ms(start_comp)
        k1, Nx1, DnZp = compute_Dnz2_opt(Nx1, DnZ.T, r_cl_ids)
        DnZ1 = DnZp.T
        start_comp = time.time_ns()
        c1 = cc_cost(k1, l, Nx1, Ny, DnZ1)
        stats.append(['COS-ROW', end_comp, c0 - c1, it])
        stats.append(['COS-ROW-CC', end_time_ms(start_comp), c0 - c1, it])
        if c0 - c1 < cost_threshold:
            no_improvements += 1
        else:
            no_improvements = 0
            c0 = c1
            k = k1
            Nx = Nx1
            DnZ = DnZ1
            Qx = Qx1
            if verbose:
                print(f'Cost after row-cluster merge: {c0}\n')

        if no_improvements >= max_no_improvements:
            break
        
        it += 1

    print(f"Final cost {c0:.6f}")
    return k, l, Nx, Ny, Qx, Qy, DnZ, c0, A, stats
