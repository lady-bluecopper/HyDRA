from helpers import compute_Dnz_opt, compute_Dnz2_opt, generate_candidate_groups_jac, end_time_ms, generate_candidate_groups
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import scipy.sparse as sp
import sys
import time
sys.path.append('.')


def compute_delta_bc(sd_cand1, sd_o, Dnzm, Dzm, den1, den2):
    term_b = 0
    term_c = 0
    tmp_cand1_blocks = list()
    if len(sd_cand1) > 0:
        densities = Dnzm[sd_cand1, :].sum(axis=1).A.flatten() / den1
        case1_mask = (densities >= 0.5)
        case2_mask = ~case1_mask
        if np.any(case1_mask):
            blks_case1 = sd_cand1[case1_mask]
            tmp_cand1_blocks.extend(blks_case1.tolist())
            corr_minus_sum_case1 = Dzm[blks_case1, -1:].sum()
            last_col_Dnzm = Dnzm.getcol(Dnzm.shape[1] - 1) 
            corr_plus_sum_case1 = last_col_Dnzm[blks_case1, :].sum() 
            term_c += (corr_minus_sum_case1 - corr_plus_sum_case1)
        if np.any(case2_mask):
            blks_case2 = sd_cand1[case2_mask]
            Dnzm_but_last_col = Dnzm[:, :-1] 
            corr_plus_sum_case2 = Dnzm_but_last_col[blks_case2, :].sum()
            Dzm_but_last_col = Dzm[:, :-1] 
            corr_minus_sum_case2 = Dzm_but_last_col[blks_case2, :].sum()
            term_c += (corr_plus_sum_case2 - corr_minus_sum_case2)
            term_b -= np.sum(case2_mask)
    if len(sd_o) > 0: 
        densities = Dnzm[sd_o, :].sum(axis=1).A.flatten() / den2
        case1_o_mask = (densities >= 0.5)
        case2_o_mask = ~case1_o_mask 
        if np.any(case1_o_mask):
            blks_case1_o = sd_o[case1_o_mask]
            tmp_cand1_blocks.extend(blks_case1_o.tolist())
            corr_minus_sum_case1_o = Dzm[blks_case1_o, :-1].sum()
            Dnzm_but_last_col_o = Dnzm[:, :-1]
            corr_plus_sum_case1_o = Dnzm_but_last_col_o[blks_case1_o, :].sum()
            term_c += (corr_minus_sum_case1_o - corr_plus_sum_case1_o)
        if np.any(case2_o_mask):
            blks_case2_o = sd_o[case2_o_mask]
            last_col_Dnzm_o = Dnzm.getcol(Dnzm.shape[1] - 1)
            corr_plus_sum_case2_o = last_col_Dnzm_o[blks_case2_o, :].sum()
            corr_minus_sum_case2_o = Dzm[blks_case2_o, -1:].sum()
            term_c += (corr_plus_sum_case2_o - corr_minus_sum_case2_o)
            term_b -= np.sum(case2_o_mask)
    return term_b, term_c, tmp_cand1_blocks


def compute_num_corrections(DnZ: csr_matrix, 
                            Dm: csr_matrix, 
                            Nx: np.ndarray, 
                            Ny: np.ndarray,
                            verbose=False) -> int:
    """
    Compute number of corrections without forming the full outer product.

    Parameters:
    ----------
    DnZ : csr_matrix
        Nonzero counts in each block.
    Dm  : csr_matrix
        Membership matrix (same shape as DnZ).
    Nx, Ny : np.ndarray
        Cluster sizes for rows (Nx) and columns (Ny).

    Returns:
    --------
    int : number of corrections
    """

    # --- Part 1: corrections for blocks selected in Dm (false positives) ---
    rows, cols = Dm.nonzero()
    block_sizes = Nx[rows] * Ny[cols]
    dnz_vals = np.array(DnZ[rows, cols]).flatten()
    corr_minus = (block_sizes - dnz_vals).sum()

    if verbose:
        print("SUM corr_minus", corr_minus)

    # --- Part 2: corrections for blocks not selected in Dm but nonzero in DnZ (false negatives) ---
    DnZ_rows, DnZ_cols = DnZ.nonzero()
    DnZ_coords = set(zip(DnZ_rows, DnZ_cols))
    Dm_coords = set(zip(rows, cols))
    target_coords = DnZ_coords - Dm_coords

    if target_coords:
        vals = np.array([DnZ[i, j] for i, j in target_coords]).flatten()
        corr_plus = vals.sum()
    else:
        corr_plus = 0

    if verbose:
        print("SUM corr_plus", corr_plus)

    return int(corr_minus + corr_plus)


def compute_num_corrections_(DnZ: csr_matrix, 
                            Dm, 
                            Bs: np.ndarray, 
                            verbose=False) -> int:
    '''
    DnZ: matrix of non-zero entries in each block
    Dm: membership matrix
    Bs: block size
    '''
    # nodes to remove
    # number of 1s in the blocks that are 0s in the original matrix
    rows, cols = Dm.nonzero() # type: ignore
    corr_minus = (Bs - DnZ)[rows, cols].sum()
    if verbose:
        print('SUM corr_minus', corr_minus)
    # nodes to add
    # number of non zero in a block where Dm does not indicate membership
    # number of 1s outside the blocks
    DnZ_rs, DnZ_cs = DnZ.nonzero()
    DnZ_coords = set(zip(DnZ_rs, DnZ_cs))
    Dm_coords = set(zip(rows, cols))
    target_coords = DnZ_coords - Dm_coords
    corr_plus = sum(DnZ[i, j] for i, j in target_coords)
    if verbose:
        print('SUM corr_plus', corr_plus)
    return corr_minus + corr_plus


def hypersummary_cost(DnZ: csr_matrix, 
                      Dm: csr_matrix, 
                      Bs: np.ndarray,
                      Ss: list[int], 
                      weights=[1, 1, 1], 
                      verbose=False):
    '''
    DnZ: matrix of non-zero entries in each block
    Dm: membership matrix
    Bs: block size
    Ss: superhyperedge sizes
    '''
    # cost of storing the weights
    e1 = len(Ss)
    # cost of storing the superhyperedges
    e2 = sum(Ss)
    # cost of corrections
    e3 = compute_num_corrections_(DnZ, Dm, Bs, verbose=verbose)
    if verbose:
        print('c(a)', e1, 'c(b)', e2, 'c(c)', e3, 'TOT', e1 + e2 + e3)
    return e1 * weights[0] + e2 * weights[1] + e3 * weights[2]


def init_cost_hs(A: csr_matrix):
    # number of superedges
    ln = A.shape[1]
    # size of superedges
    nnz = A.count_nonzero()
    # corrections
    c = 0
    return ln + nnz + c


def cc_cost_hs(NX: np.ndarray, 
               NY: np.ndarray, 
               DnZ: csr_matrix, 
               weights=[1, 1, 1], 
               verbose=False):
    """
    Parameters:
    Nx, Ny : Arrays of row and column cluster sizes.
    DnZ    : csr_matrix of nonzero counts per block.

    Returns:
    c      : Total encoding cost.
    """
    Nx = NX.copy()
    Ny = NY.copy()

    # Step 0: Disregard empty clusters
    reduce = False
    non_empty_row_clusters = np.arange(len(Nx))
    non_empty_col_clusters = np.arange(len(Ny))
    if np.any(Nx == 0):
        reduce = True
        non_empty_row_clusters = np.nonzero(Nx)[0]
    if np.any(Ny == 0):
        reduce = True
        non_empty_col_clusters = np.nonzero(Ny)[0]
    if reduce:
        Nx = Nx[non_empty_row_clusters]
        Ny = Ny[non_empty_col_clusters]
        Dnz = DnZ[np.ix_(non_empty_row_clusters, non_empty_col_clusters)]
    else:
        Dnz = DnZ
    # cost of storing the weights
    e1 = len(non_empty_col_clusters)

    # ---- FIX: avoid np.outer(Nx, Ny) ----
    rows, cols = Dnz.nonzero()
    vals = np.array(Dnz[rows, cols]).flatten()

    # normalize only on nonzero positions
    normed = vals / (Nx[rows] * Ny[cols])
    mask = normed >= 0.5

    memberships = csr_matrix(
        (mask.astype(int), (rows, cols)),
        shape=Dnz.shape
    )
    # -------------------------------------

    # cost of storing the superhyperedges
    e2 = memberships.sum()
    # cost of corrections
    e3 = compute_num_corrections(Dnz, memberships, 
                                 Nx, Ny, 
                                 verbose=verbose)
    if verbose:
        print('c(a)', e1, 'c(b)', e2, 'c(c)', e3, 'TOT', e1 + e2 + e3)
    return e1 * weights[0] + e2 * weights[1] + e3 * weights[2]


def init_perm_hs(r: int, 
                 b: int, 
                 A: csr_matrix, 
                 weights=[1, 1, 1], 
                 verbose=False):
    m, n = A.shape
    Nx = np.ones(m, dtype=np.int64)
    Ny = np.ones(n, dtype=np.int64)
    Qx = np.arange(m)
    Qy = np.arange(n)
    if verbose:
        print('First Round on Columns')
    Ny, Qy = jac_col_hash2_hs(A, r, b, Ny, Qy,
                              col_merge=True,
                              weights=weights,
                              verbose=verbose)
    if verbose:
        print('First Round on Rows')
    Nx, Qx = jac_col_hash2_hs(A.T, r, b, Nx, Qx,
                              col_merge=False,
                              weights=weights,
                              verbose=verbose)
    k, ln, Nx, Ny, DnZ = compute_Dnz_opt(Qx, Nx, Qy, Ny, A)
    return k, ln, Nx, Ny, Qx, Qy, DnZ


def symm_diff_and_inter(lst1, lst2):
    '''
    Symmetric difference and intersection
    between lst1 and lst2.
    '''
    inter = []
    sd1 = []
    sd2 = []
    s1 = set(lst1)
    s2 = set(lst2)
    for i in s1:
        if i in s2:
            s2.remove(i)
            inter.append(i)
        else:
            sd1.append(i)
    for i in s2:
        sd2.append(i)
    return sd1, sd2, inter


def cos_col_hash_hs(r: int, 
                    b: int,
                    k0: int, 
                    l0: int,
                    Nx: np.ndarray, 
                    NY: np.ndarray,
                    QY: np.ndarray,
                    DnZ: csr_matrix,
                    col_merge: bool=True,
                    min_size: int=2,
                    max_trials: int=5,
                    weights=[1, 1, 1],
                    verbose=False):
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

    cluster_ids = np.unique(Qy)
    assert len(cluster_ids) == len(Ny)
    assert len(cluster_ids) == np.sum(Ny > 0)
    assert len(cluster_ids) == l0
    
    ucg, candidategroups = generate_candidate_groups(r, b, k0, l0, Pnz)

    # Step 4: Merge candidate groups
    for enid, uc in enumerate(ucg):
        # all (groups of) columns/rows in some group with uc
        ind = np.where(candidategroups == uc)[0]
        num = len(ind)
        if num < min_size:
            continue

        trials = 0
        while True:
            merged = False
            # take only non-empty clusters
            ix = np.where(Ny[ind] > 0)[0]
            if len(ix) == 0:
                break
            # pick random element from the group ind
            fo = np.random.choice(ix)
            cand1 = [ind[fo]]
            Dm = csc_matrix((Pnz >= 0.5).astype(int))
            # row (col) clusters assigned to col (row) cluster cand1
            cand1_blks = Dm.indices[Dm.indptr[ind[fo]]:Dm.indptr[ind[fo] + 1]]
            Dnzc = Dnz.getcol(cand1[0])
            Pnzc = Pnz[:, cand1[0]].copy()
            Dzc = Dz[:, cand1[0]].copy()
            # for all the other elements
            for o in range(num):
                if ind[o] in cand1 or Ny[ind[o]] == 0:
                    continue
                Dnzm = sp.hstack((Dnzc, Dnz.getcol(ind[o]))).tocsr()
                Dzm = np.hstack([Dzc, Dz[:, ind[o]:ind[o] + 1]])
                cis = [np.sum(Ny[cand1]), Ny[ind[o]]]
                cis_sum = np.sum(cis)

                Dnzv = Dnzm.sum(axis=1)
                newrowsizes = np.dot(Nx, sum(cis)).reshape(-1, 1)
                Pnzv = Dnzv / newrowsizes
                Dzv = newrowsizes - Dnzv

                # row (col) clusters assigned to col (row) cluster ind[o]
                o_blks = Dm.indices[Dm.indptr[ind[o]]:Dm.indptr[ind[o] + 1]]
                # clusters in cand1 not in ind[o],
                # clusters in ind[o] not in cand1
                # clusters in both cand1 and ind[o]
                sd_cand1, sd_o, inter_cand1o = symm_diff_and_inter(cand1_blks, o_blks)
                sd_cand1 = np.array(sd_cand1, dtype=int)
                sd_o = np.array(sd_o, dtype=int)
                tmp_cand1_blocks = inter_cand1o

                term_b = -len(inter_cand1o)
                # term_c = 0
                if col_merge:
                    # COST a
                    term_a = -1  # -1 weights to store
                else:
                    term_a = 0  # number of super-hyperedges does not change
                # COST b and c
                den1 = 0
                den2 = 0
                if len(sd_cand1) > 0:
                    den1 = cis_sum * Nx[sd_cand1]
                if len(sd_o) > 0:
                    den2 = cis_sum * Nx[sd_o]

                d_tb, term_c, new_blocks = compute_delta_bc(sd_cand1, sd_o, Dnzm, Dzm, den1, den2)
                term_b += d_tb
                tmp_cand1_blocks.extend(new_blocks)

                cur_cost = term_a * weights[0] + term_b * weights[1] + term_c * weights[2]
                if cur_cost <= 0:
                    cand1_blks = tmp_cand1_blocks
                    merged = True
                    cand1.append(ind[o])
                    Dnzc = Dnzv
                    Pnzc = Pnzv
                    Dzc = Dzv

            if len(cand1) > 1:

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
                Pnz[:, newind[minind]: newind[minind] + 1] = Pnzc
                Dz[:, newind[minind]] = Dzc

            if not merged:
                trials += 1
            if r == 1:
                if trials > max_trials:
                    break
            elif trials > 0:
                break
    return Ny, Qy, cluster_ids


def jac_col_hash2_hs(A: csr_matrix, 
                     r: int, 
                     b: int, 
                     NY: np.ndarray, 
                     QY: np.ndarray, 
                     col_merge: bool,
                     min_size: int=5,
                     weights=[1, 1, 1],
                     verbose: bool=False):
    '''
    r: code length
    b: num hash tables
    NY: sizes of column clusters
    QY: cluster ids of columns
    '''
    Ny = NY.copy()
    Qy = QY.copy()
    
    ucg, candidategroups = generate_candidate_groups_jac(r, b, A)
    if verbose:
        print('num candidate groups', len(ucg))
    A_csc = csc_matrix(A)

    # try merging candidate GROUPS, don't if cost larger
    for uc in ucg:
        ind = np.where(candidategroups == uc)[0]
        num = len(ind)
        if num < min_size:
            continue
        # try to merge 1 by 1
        cand1 = [ind[0]]
        cand1_blocks = A_csc.indices[A_csc.indptr[ind[0]]:
                                     A_csc.indptr[ind[0] + 1]]
        Dnzc = A_csc.getcol(cand1[0])
        Dzc = csr_matrix(np.ones(Dnzc.shape) - Dnzc)

        for o in range(1, num):
            # merge group-cand1 with group ind(o)?
            cis = np.array([np.sum(Ny[cand1]), 1])
            sum_cis = np.sum(cis)
            # reduce cluster assignment cost
            Ac = A_csc.getcol(ind[o])
            Dnzm = sp.hstack((Dnzc, Ac)).tocsr()
            Dzm = sp.hstack((Dzc, np.ones(Ac.shape) - Ac)).tocsr()
            Dnzv = Dnzm.sum(axis=1)
            Dzv = sum_cis - Dnzv

            # row (col) clusters assigned to col (row) cluster ind[o]
            o_blocks = A_csc.indices[A_csc.indptr[ind[o]]:
                                     A_csc.indptr[ind[o] + 1]]
            # clusters in cand1 not in ind[o],
            # clusters in ind[o] not in cand1
            # clusters in both cand1 and ind[o]
            sd_cand1, sd_o, inter_cand1o = symm_diff_and_inter(cand1_blocks,
                                                               o_blocks)
            sd_cand1 = np.array(sd_cand1, dtype=int)
            sd_o = np.array(sd_o, dtype=int)
            tmp_cand1_blocks = inter_cand1o

            term_b = -len(inter_cand1o)
            if col_merge:
                # COST a
                term_a = -1  # -1 weights to store
            else:
                term_a = 0  # number of super-hyperedges does not change
            # COST b and c
            d_tb, term_c, new_blocks = compute_delta_bc(sd_cand1, sd_o, Dnzm, Dzm, sum_cis, sum_cis)
            term_b += d_tb
            tmp_cand1_blocks.extend(new_blocks)
                
            cur_cost = term_a * weights[0] + term_b * weights[1] + term_c * weights[2]
            if verbose:
                print('cur_cost', cur_cost)
                print('term_a', term_a, 'term_b', term_b, 'term_c', term_c)
            if cur_cost <= 0:
                cand1_blocks = tmp_cand1_blocks
                cand1.append(ind[o])
                Dnzc = Dnzv
                Dzc = csr_matrix(Dzv)

        if len(cand1) > 1:
            if verbose:
                print('merging', cand1)
            newind = cand1
            # Ny and Qy are going to change
            Ny[newind] = 0
            Ny[min(newind)] = len(newind)
            Qy[newind] = min(newind)

    return Ny, Qy


def hc_search_hs(A, 
                 r=20, 
                 b=5, 
                 seed=42, 
                 weights=[1, 1, 1],
                 cost_threshold: float=1,
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
    np.random.seed(seed)
    # Initialize costs and variables
    cinit = init_cost_hs(A)
    # Initialize structures to store results
    start_comp = time.time_ns()
    stats = []
    if verbose:
        print(f"Starting cost is {cinit:.6f}.")

    k, ln, Nx, Ny, Qx, Qy, DnZ = init_perm_hs(r, b, A,
                                              weights=weights,
                                              verbose=verbose)
    end_comp = end_time_ms(start_comp)
    start_comp = time.time_ns()
    c0 = cc_cost_hs(Nx, Ny, DnZ, weights=weights, verbose=verbose)
    stats.append(['JAC', end_comp, cinit - c0, 0])
    stats.append(['JAC-CC', end_time_ms(start_comp), cinit - c0, 0])
    if verbose:
        print(f"Cost after Jaccard-based iteration: {c0:.6f}.")

    no_improvements = 0
    it = 1
    r1 = r
    b1 = b
    C = 1.2
    while True:
        r2 = round(r1 / (C**(it - 1)))
        b2 = round(b1 * (C**(it - 1)))
        if verbose:
            print('round', it, 'r=', r2, 'b=', b2)
        start_comp = time.time_ns()
        Ny1, Qy1, c_cl_ids = cos_col_hash_hs(r2, b2, k, ln,
                                             Nx, Ny, Qy, DnZ,
                                             col_merge=True,
                                             weights=weights,
                                             min_size=min_size,
                                             max_trials=max_trials,
                                             verbose=verbose)
        end_comp = end_time_ms(start_comp)
        l1, Ny1, DnZ1 = compute_Dnz2_opt(Ny1, DnZ, c_cl_ids)
        start_comp = time.time_ns()
        c1 = cc_cost_hs(Nx, Ny1, DnZ1, weights=weights, verbose=verbose)
        stats.append(['COS-COL-rb', end_comp, c0 - c1, it])
        stats.append(['COS-COL-rb-CC', end_time_ms(start_comp), c0 - c1, it])
        cthresh = cost_threshold
        if cost_threshold < 1:
            cthresh *= c0
        if c0 - c1 < cthresh:
            no_improvements += 1
        else:
            no_improvements = 0
            c0 = c1
            ln = l1
            Ny = Ny1
            DnZ = DnZ1
            Qy = Qy1
            if verbose:
                print(f"Cost after col-cluster merge: {c0:.6f}")
        start_comp = time.time_ns()
        Nx1, Qx1, r_cl_ids = cos_col_hash_hs(r2, b2, ln, k,
                                             Ny, Nx, Qx, DnZ.T,
                                             col_merge=False,
                                             weights=weights,
                                             min_size=min_size,
                                             max_trials=max_trials,
                                             verbose=verbose)
        end_comp = end_time_ms(start_comp)
        k1, Nx1, DnZp = compute_Dnz2_opt(Nx1, DnZ.T, r_cl_ids)
        DnZ1 = DnZp.T
        start_comp = time.time_ns()
        c1 = cc_cost_hs(Nx1, Ny, DnZ1, weights=weights, verbose=verbose)
        stats.append(['COS-ROW-rb', end_comp, c0 - c1, it])
        stats.append(['COS-ROW-rb-CC', end_time_ms(start_comp), c0 - c1, it])
        cthresh = cost_threshold
        if cost_threshold < 1:
            cthresh *= c0
        if c0 - c1 < cthresh:
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

        if verbose:
            print(f'Iteration with r={1}, b={100}')
        start_comp = time.time_ns()
        Ny1, Qy1, c_cl_ids = cos_col_hash_hs(1, 100, k, ln, Nx, Ny,
                                             Qy, DnZ, col_merge=True,
                                             weights=weights,
                                             min_size=min_size,
                                             max_trials=max_trials,
                                             verbose=verbose)
        end_comp = end_time_ms(start_comp)
        l1, Ny1, DnZ1 = compute_Dnz2_opt(Ny1, DnZ, c_cl_ids)
        start_comp = time.time_ns()
        c1 = cc_cost_hs(Nx, Ny1, DnZ1, weights=weights, verbose=verbose)
        stats.append(['COS-COL', end_comp, c0 - c1, it])
        stats.append(['COS-COL-CC', end_time_ms(start_comp), c0 - c1, it])
        cthresh = cost_threshold
        if cost_threshold < 1:
            cthresh *= c0
        if c0 - c1 < cthresh:
            no_improvements += 1
        else:
            no_improvements = 0
            c0 = c1
            ln = l1
            Ny = Ny1
            DnZ = DnZ1
            Qy = Qy1
            if verbose:
                print(f'Cost after col-cluster merge: {c0}\n')

        start_comp = time.time_ns()
        Nx1, Qx1, r_cl_ids = cos_col_hash_hs(1, 100, ln, k, Ny, Nx,
                                             Qx, DnZ.T,
                                             col_merge=False,
                                             weights=weights,
                                             min_size=min_size,
                                             max_trials=max_trials,
                                             verbose=verbose)
        end_comp = end_time_ms(start_comp)
        k1, Nx1, DnZp = compute_Dnz2_opt(Nx1, DnZ.T, r_cl_ids)
        DnZ1 = DnZp.T
        start_comp = time.time_ns()
        c1 = cc_cost_hs(Nx1, Ny, DnZ1, weights=weights, verbose=verbose)
        stats.append(['COS-ROW', end_comp, c0 - c1, it])
        stats.append(['COS-ROW-CC', end_time_ms(start_comp), c0 - c1, it])
        cthresh = cost_threshold
        if cost_threshold < 1:
            cthresh *= c0
        if c0 - c1 < cthresh:
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
    return k, ln, Nx, Ny, Qx, Qy, DnZ, c0, A, stats
