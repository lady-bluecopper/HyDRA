import numpy as np
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt


def cc_label_perm(k, Qx):

    u = np.unique(Qx)
    Px = np.zeros(len(Qx))

    start = 0
    for ik in range(k):
        idxs_k = np.where(Qx == u[ik])[0]
        len_k = len(idxs_k)
        Px[start:start + len_k] = idxs_k
        start = start + len_k
    return Px


def cc_perm(Qx, Qy):
    Px = cc_label_perm(len(np.unique(Qx)), Qx)
    Py = cc_label_perm(len(np.unique(Qy)), Qy)
    return Px, Py


def plot_binary_duo(A, nx, ny, Nx, Ny, fig_path,
                    ratio=1, row_mask=None, col_mask=None):
    """
    Plot a binary matrix (optionally masking out select rows and columns).

    Parameters:
        A (ndarray): Binary matrix with values 0/1.
        is_grouped (bool): Whether the plot represents grouped data.
        ratio (float): Aspect ratio for non-square plots. Default is 1.
        row_mask (ndarray): Boolean mask for rows. Default is no masking.
        col_mask (ndarray): Boolean mask for columns. Default is no masking.
    """
    if row_mask is None:
        row_mask = np.zeros(A.shape[0], dtype=bool)
    if col_mask is None:
        col_mask = np.zeros(A.shape[1], dtype=bool)

    is_self_graph = A.shape[0] == A.shape[1]

    # Create a temporary matrix to represent masked and unmasked values
    Atmp = deepcopy(A).todense().astype(float)
    # Mark masked rows and columns with a special value
    Atmp[np.ix_(row_mask, col_mask)] = 2

    cmap = matplotlib.colors.ListedColormap([
        (0.3, 0.3, 0.3),  # Gray for zeros
        (0.0, 0.0, 0.0),  # Black for masked
        (1, 0.6, 0.8)   # Light red for ones
    ])

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(Atmp, aspect='auto', cmap=cmap, origin='lower',
              extent=(0, ny, 0, nx), interpolation='nearest')
    ax.set_ylabel('Left Nodes', fontsize=14)
    ax.set_xlabel('Right Nodes', fontsize=14)

    if is_self_graph:
        ax.set_aspect(ratio)
    # horizonal lines
    for x in np.cumsum(Nx):
        ax.axhline(x, color='r', linestyle='--', linewidth=1)
    # vertical lines
    for y in np.cumsum(Ny):
        ax.axvline(y, color='r', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def plot_matrix(A, nx, ny, ratio=1):
    """
    Plot a binary matrix.

    Parameters:
        A (ndarray): Binary matrix with values 0/1.
        ratio (float): Aspect ratio for non-square plots. Default is 1.
    """
    is_self_graph = A.shape[0] == A.shape[1]

    # Create a temporary matrix to represent masked and unmasked values
    Atmp = deepcopy(A).todense().astype(float)

    cmap = matplotlib.colors.ListedColormap([
        (0.3, 0.3, 0.3),  # Gray for zeros
        (0.0, 0.0, 0.0),  # Black for masked
        (1, 0.6, 0.8)   # Light red for ones
    ])

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(Atmp, aspect='auto', cmap=cmap, origin='lower',
              extent=(0, ny, 0, nx), interpolation='nearest')
    ax.set_ylabel('Left Nodes', fontsize=14)
    ax.set_xlabel('Right Nodes', fontsize=14)

    if is_self_graph:
        ax.set_aspect(ratio)
    plt.show()
