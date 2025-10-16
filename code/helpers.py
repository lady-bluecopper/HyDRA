from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp
import time


def compute_Dnz2_opt(NY: np.ndarray,
                     DNZ: csr_matrix,
                     cluster_ids: np.ndarray):
    """
    Compute DnZ.

    Parameters:
        Ny (numpy array): Array of cluster sizes.
        DnZ (numpy array): Matrix of non-zeros in each block.
        cluster_ids (numpy array): Cluster identifiers.

    Returns:
        ln (int): Number of column clusters.
        Ny (numpy array): Updated column cluster sizes.
        DnZ (numpy array): Updated matrix of non-zeros in each block.
    """
    cl_ids_c = cluster_ids.copy()
    Ny = NY.copy()

    uy = np.unique(cl_ids_c)
    # Filter Ny for non-zero elements
    Ny = Ny[Ny > 0]
    ln = len(Ny)
    assert len(uy) == ln

    # --- Process columns (Y clusters) ---
    # Create a mapping matrix for column aggregation
    # M_col will map original columns to the new, unique columns.
    # Original columns that map to the same unique column are summed.
    num_original_cols = DNZ.shape[1]
    col_mapping_rows = []
    col_mapping_cols = []
    col_mapping_data = []
    for j in range(ln):
        # Find all original column indices that belong to this unique cluster
        ind = np.where(cl_ids_c == uy[j])[0]
        for original_col_idx in ind:
            col_mapping_rows.append(j)
            col_mapping_cols.append(original_col_idx)
            col_mapping_data.append(1)
    # Construct the sparse transformation matrix T_col (l x num_original_cols)
    # T_col * DnZ' will aggregate columns
    T_col = coo_matrix((col_mapping_data, (col_mapping_rows,
                                           col_mapping_cols)),
                        shape=(ln, num_original_cols), dtype=DNZ.dtype).tocsr()
    # Apply column aggregation: DnZ_new = DnZ @ T_col.T
    DnZ = (T_col * DNZ.T).T

    return ln, Ny, DnZ


def compute_Dnz_opt(Qx: np.ndarray,
                    NX: np.ndarray,
                    Qy: np.ndarray,
                    NY: np.ndarray,
                    DNZ: csr_matrix):

    ux = np.unique(Qx)
    uy = np.unique(Qy)

    Qxc = Qx.copy()
    Qyc = Qy.copy()

    Nx = NX.copy()
    Ny = NY.copy()

    Nx = Nx[ux]
    Ny = Ny[uy]

    k = len(Nx)
    ln = len(Ny)

    # --- Process columns (Y clusters) ---
    # Create a mapping matrix for column aggregation
    # M_col will map original columns to the new, unique columns.
    # Original columns that map to the same unique column are summed.
    num_original_cols = DNZ.shape[1]
    col_mapping_rows = []
    col_mapping_cols = []
    col_mapping_data = []

    for j in range(ln):
        # Find all original column indices that belong to this unique cluster
        ind = np.where(Qyc == uy[j])[0]
        for original_col_idx in ind:
            col_mapping_rows.append(j)
            col_mapping_cols.append(original_col_idx)
            col_mapping_data.append(1)
    # Construct the sparse transformation matrix T_col (l x num_original_cols)
    # T_col * DnZ' will aggregate columns
    T_col = coo_matrix((col_mapping_data, (col_mapping_rows,
                                           col_mapping_cols)),
                        shape=(ln, num_original_cols), dtype=DNZ.dtype).tocsr()
    # Apply column aggregation: DnZ_new = DnZ @ T_col.T
    DnZ = (T_col * DNZ.T).T
    assert DnZ.shape[1] == ln

    # --- Process rows (X clusters) ---
    # Create a mapping matrix for row aggregation
    num_original_rows = DNZ.shape[0]
    row_mapping_rows = []
    row_mapping_cols = []
    row_mapping_data = []

    for i in range(k):
        # Find all original row indices that belong to this unique cluster
        ind = np.where(Qxc == ux[i])[0]
        for original_row_idx in ind:
            row_mapping_rows.append(i)
            row_mapping_cols.append(original_row_idx)
            row_mapping_data.append(1)
    # Construct the sparse transformation matrix T_row (k x num_original_rows)
    T_row = coo_matrix((row_mapping_data, (row_mapping_rows,
                                           row_mapping_cols)),
                        shape=(k, num_original_rows), dtype=DNZ.dtype).tocsr()

    # Apply row aggregation: DnZ_new = T_row @ DnZ_c
    DnZ = T_row @ DnZ
    assert DnZ.shape[0] == k

    return k, ln, Nx, Ny, DnZ


def generate_candidate_groups(r: int,  # signature size
                              b: int,  # num hash tables
                              k0: int,  # num row clusters
                              l0: int,  # num col clusters
                              Pnz):  # probability non-zero
    # Step 1: Generate signatures (COSINE SIMILARITY)
    s = r * b  # Total number of signatures
    S = np.zeros((s, l0))  # Signature matrix
    for i in range(s):
        ri = np.random.randn(k0)  # pick hyperplane
        ri /= np.sqrt(np.sum(ri**2))
        S[i, :] = np.sign(ri @ Pnz)
    # Step 2: Generate b hash tables
    maps = []
    for j in range(b):
        start = r * j
        end = start + r
        c = defaultdict(list)
        for i in range(l0):
            t = " ".join([str(x) for x in S[start:end, i]])
            c[t].append(i)
        maps.append(c)
    # Step 3: Candidate groups
    candidategroups = np.arange(l0)
    for c in maps:  # for each hash table
        # (groups of) columns/rows hashed to the same bucket
        for group in c.values():
            min_group = min(group)  # representative of the bucket
            for idx in group:
                candidategroups[idx] = min_group
    # unique representatives (a column/row may represent more than 1 group)
    ucg = np.unique(candidategroups)
    return ucg, candidategroups


def generate_candidate_groups_jac(r: int,  # signature size
                                  b: int,  # num hash tables
                                  A):  # adjacency matrix
    s = r * b
    n, m = A.shape
    # generate permutations
    M = np.zeros((s, n))
    for i in range(s):
        M[i, :] = np.random.permutation(n)
    # generate signatures
    S = np.zeros((s, m))
    for i in range(m):
        if A[:, i].nnz > 0:
            # Step 1: Create the boolean mask where A(:, i) > 0
            mask = A[:, i].nonzero()[0]
            # Step 2: Apply the mask to M and compute the min along rows
            S[:, i] = np.min(M[:, mask], axis=1)
    # create b hash-tables
    maps = []
    for j in range(b):
        start = r * j
        end = start + r
        c = defaultdict(list)
        # hash all columns to hash-table j
        for i in range(m):
            t = " ".join([str(x) for x in S[start:end, i]])
            c[t].append(i)
        maps.append(c)
    # now trying-to-merge-phase
    candidategroups = np.arange(m)
    # process hash-tables 1 by 1
    for c in maps:
        for group in c.values():
            min_group = min(group)
            for idx in group:
                candidategroups[idx] = min_group
    ucg = np.unique(candidategroups)
    return ucg, candidategroups


def logstar2(n):
    if n <= 0:
        return 0
    lg = 0
    n = np.log2(n)
    while n > 0:
        lg += n
        n = np.log2(n)
    lg += np.log2(2.8665064)
    return lg


def entropy_bits(v):
    if isinstance(v, (int, float)):
        L = v
    else:
        L = v.copy()
        if sp.issparse(L):
            L = L.toarray()
    L += np.exp(-700)
    return - np.log2(L)


def contains_zero(a):
    for v in a:
        if v == 0:
            return True
    return False


def end_time_ms(st):
    return (time.time_ns() - st) / 1_000_000
