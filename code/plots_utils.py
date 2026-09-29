import numpy as np
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd


def cc_label_perm(k, Qx):
    
    u = np.unique(Qx)
    Px = np.zeros(len(Qx))

    start = 0
    for ik in range(k):
        idxs_k = np.where(Qx == u[ik])[0]
        len_k = len(idxs_k)
        Px[start:start+len_k] = idxs_k
        start = start + len_k
    return Px


def cc_perm(Qx, Qy):
    Px = cc_label_perm(len(np.unique(Qx)), Qx)
    Py = cc_label_perm(len(np.unique(Qy)), Qy)
    return Px, Py


def plot_binary_duo(A, nx, ny, Nx, Ny, fig_path, ratio=1, row_mask=None, col_mask=None):
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
    Atmp[np.ix_(row_mask, col_mask)] = 2  # Mark masked rows and columns with a special value
    
    cmap = matplotlib.colors.ListedColormap([
        (0.3, 0.3, 0.3),  # Gray for zeros
        (0.0, 0.0, 0.0),  # Black for masked
        (1, 0.6, 0.8)   # Light red for ones
    ])
    
    fig, ax = plt.subplots(figsize=(10,10))
    
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
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    ax.imshow(Atmp, aspect='auto', cmap=cmap, origin='lower', 
              extent=(0, ny, 0, nx), interpolation='nearest')
    ax.set_ylabel('Left Nodes', fontsize=14)
    ax.set_xlabel('Right Nodes', fontsize=14)

    if is_self_graph:
        ax.set_aspect(ratio)
    plt.show()


def plot_correction_table_distribution(
        summary_dir,
        algorithm,
        dataset_order=None,
        dataset_labels=None,
        save_path=None,
        colors=("#4C78A8", "#F58518", "#B8B8B8", "#7A5195")):
    """Measure and plot correction-table sizes across datasets.

    The left panel decomposes the correction-table cost into additions
    (``C+``), removals (``C-``), and optional stored weights, normalized by
    the cost of the original hypergraph.  The right panel reports the share
    of the *summary* cost occupied by its correction table.  Bars are means
    across runs; error bars in the right panel show one standard deviation.

    Parameters
    ----------
    summary_dir : str or os.PathLike
        Directory containing ``summary__*.csv`` and ``original__*.csv``.
    algorithm : str
        Exact value stored in the ``algo=...`` part of summary filenames,
        for example ``"Leman(our)"`` or ``"HyDRA"``.
    dataset_order : sequence of str, optional
        Dataset order.  When omitted, datasets are sorted alphabetically.
    dataset_labels : mapping or callable, optional
        Display labels keyed by dataset name, or a function returning one.
    save_path : str or os.PathLike, optional
        If supplied, save the figure at this path.

    Returns
    -------
    stats : pandas.DataFrame
        One row per summary/run with raw counts and normalized measures.
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray
        The two subplot axes.
    """
    # Import lazily so this general plotting module remains lightweight.
    import summary_utils as sut

    summary_dir = Path(summary_dir)
    if not summary_dir.is_dir():
        raise FileNotFoundError(f"summary directory does not exist: {summary_dir}")

    requested = None if dataset_order is None else list(dataset_order)
    requested_set = None if requested is None else set(requested)
    rows = []
    for summary_path in sorted(summary_dir.glob("summary__*.csv")):
        params = sut.get_params_from_summary_name(summary_path.name)
        dataset = params.get("data")
        if params.get("algo") != algorithm:
            continue
        if requested_set is not None and dataset not in requested_set:
            continue

        _, superedges, _, corrections, _ = sut.load_summary(
            str(summary_path), {}
        )
        original_path = summary_dir / f"original__data={dataset}.csv"
        if not original_path.exists():
            raise FileNotFoundError(
                f"original hypergraph encoding not found: {original_path}"
            )
        _, original_edges, _, _, _ = sut.load_summary(str(original_path), {})

        additions = 0
        removals = 0
        stored_weights = 0
        records = 0
        for correction_list in corrections.values():
            records += len(correction_list)
            for correction in correction_list:
                additions += len(correction[0])
                removals += len(correction[1])
                stored_weights += int(len(correction) > 2)

        superedge_weights = len(superedges)
        superedge_memberships = sum(len(edge) for edge in superedges)
        correction_cost = additions + removals + stored_weights
        summary_cost = (
            superedge_weights + superedge_memberships + correction_cost
        )
        original_cost = len(original_edges) + sum(
            len(edge) for edge in original_edges
        )
        if original_cost <= 0 or summary_cost <= 0:
            raise ValueError(
                f"non-positive encoding cost for {summary_path.name}"
            )

        rows.append({
            "Dataset": dataset,
            "Algorithm": algorithm,
            "Run": int(params.get("run", 0)),
            "Summary Path": str(summary_path),
            "Correction Records": records,
            "C+ Entries": additions,
            "C- Entries": removals,
            "Stored Weight Fields": stored_weights,
            "Correction Cost": correction_cost,
            "Superhyperedge Weight Cost": superedge_weights,
            "Superhyperedge Membership Cost": superedge_memberships,
            "Summary Cost": summary_cost,
            "Original Cost": original_cost,
            "C+ / Original Cost": additions / original_cost,
            "C- / Original Cost": removals / original_cost,
            "Weight Fields / Original Cost": stored_weights / original_cost,
            "Correction / Original Cost": correction_cost / original_cost,
            "Correction Share": correction_cost / summary_cost,
        })

    if not rows:
        raise FileNotFoundError(
            f"no summaries for algorithm {algorithm!r} in {summary_dir}"
        )

    stats = pd.DataFrame(rows)
    observed = list(stats["Dataset"].unique())
    order = requested if requested is not None else sorted(observed)
    order = [dataset for dataset in order if dataset in observed]
    grouped = stats.groupby("Dataset", sort=False)
    means = grouped.mean(numeric_only=True).reindex(order)
    standard_deviations = grouped.std(numeric_only=True).reindex(order).fillna(0)

    if dataset_labels is None:
        labels = order
    elif callable(dataset_labels):
        labels = [dataset_labels(dataset) for dataset in order]
    else:
        labels = [dataset_labels.get(dataset, dataset) for dataset in order]

    fig, axes = plt.subplots(1, 2, figsize=(max(9, 1.25 * len(order)), 4.2))
    positions = np.arange(len(order))
    bottom = np.zeros(len(order), dtype=float)
    components = (
        ("C+ / Original Cost", r"$C^+$ entries", colors[0]),
        ("C- / Original Cost", r"$C^-$ entries", colors[1]),
        ("Weight Fields / Original Cost", "stored weights", colors[2]),
    )
    for column, label, color in components:
        values = 100 * means[column].to_numpy()
        axes[0].bar(positions, values, bottom=bottom, label=label, color=color)
        bottom += values

    share = 100 * means["Correction Share"].to_numpy()
    share_sd = 100 * standard_deviations["Correction Share"].to_numpy()
    axes[1].bar(positions, share, color=colors[3], yerr=share_sd, capsize=3)

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.set_xlabel("")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Correction cost / original cost (%)")
    axes[0].set_title("Lossless correction overhead")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Correction share of summary cost (%)")
    axes[1].set_title("Composition of the stored summary")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.parent != Path(""):
            os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return stats, fig, axes
