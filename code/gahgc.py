"""Independent GAHGC reimplementation for the HyDRA experiment framework.

The implementation follows Wang, Wang, and Li (IEEE TCSS 2026) where the
paper is unambiguous: features are hyperedges, Equation 9 supplies their
weights, weighted hypergraph walks provide sample contexts, and column labels
are the dominant row label of their incident samples.

Equations 18--20 do not define an implementable center update.  We therefore
use an objective-faithful solver for Equation 7: ordinary embedding centers
alternate with ICM assignments minimizing squared distance plus weighted
within-hyperedge label disagreement.  This is an independent reimplementation,
not official author code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


Matrix = np.ndarray | sparse.spmatrix


def _as_binary_csr(x: Matrix | ArrayLike) -> sparse.csr_matrix:
    """Return topology-preserving binary incidence without densifying."""
    if sparse.issparse(x):
        out = sparse.csr_matrix(x, dtype=np.float64)
    else:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")
        out = sparse.csr_matrix(values)
    if out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError("X must have at least one row and one column")
    if not np.isfinite(out.data).all():
        raise ValueError("X contains NaN or infinity")
    out.eliminate_zeros()
    out.data = np.ones(out.nnz, dtype=np.float64)
    empty_columns = np.flatnonzero(np.diff(out.tocsc().indptr) == 0)
    if empty_columns.size:
        raise ValueError("GAHGC requires nonempty hyperedges/columns")
    return out


def _sparse_row_sum(x: sparse.csr_matrix, rows: np.ndarray) -> sparse.csr_matrix:
    result = x.getrow(int(rows[0])).copy()
    for row in rows[1:]:
        result = result + x.getrow(int(row))
    return sparse.csr_matrix(result)


def _edge_distance_squares(
    x: sparse.csr_matrix,
    vertices: np.ndarray,
    row_norms: np.ndarray,
) -> np.ndarray:
    q = int(vertices.size)
    if q <= 1:
        return np.zeros(q, dtype=float)
    summed = _sparse_row_sum(x, vertices)
    dot = np.asarray(x[vertices].dot(summed.T).toarray()).ravel()
    summed_norm = float(summed.multiply(summed).sum())
    numerator = q * q * row_norms[vertices] - 2.0 * q * dot + summed_norm
    return np.maximum(numerator / ((q - 1) ** 2), 0.0)


def _hyperedge_weights(
    x: sparse.csr_matrix,
    b_csc: sparse.csc_matrix,
    bandwidth: float | Literal["median"],
) -> tuple[np.ndarray, float]:
    row_norms = np.asarray(x.multiply(x).sum(axis=1)).ravel()
    if bandwidth == "median":
        samples: list[np.ndarray] = []
        remaining = 4096
        for edge in range(b_csc.shape[1]):
            vertices = b_csc.indices[b_csc.indptr[edge] : b_csc.indptr[edge + 1]]
            if vertices.size > 1:
                distances = _edge_distance_squares(x, vertices, row_norms)
                take = min(remaining, distances.size)
                samples.append(distances[:take])
                remaining -= take
                if remaining <= 0:
                    break
        positive = np.concatenate(samples) if samples else np.array([1.0])
        positive = positive[positive > 0]
        delta = float(np.median(positive)) if positive.size else 1.0
    else:
        delta = float(bandwidth)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("bandwidth must be positive, finite, or 'median'")

    weights = np.zeros(b_csc.shape[1], dtype=float)
    for edge in range(b_csc.shape[1]):
        vertices = b_csc.indices[b_csc.indptr[edge] : b_csc.indptr[edge + 1]]
        q = int(vertices.size)
        if q == 1:
            weights[edge] = 1.0
        elif q > 1:
            distances = _edge_distance_squares(x, vertices, row_norms)
            # Literal Equation 9 normalization: q terms divided by q - 1.
            weights[edge] = float(np.exp(-distances / delta).sum() / (q - 1))
    return weights, delta


def _sample_walks(
    b_csr: sparse.csr_matrix,
    b_csc: sparse.csc_matrix,
    weights: np.ndarray,
    n_walks: int,
    max_length: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    walks: list[np.ndarray] = []
    for start in range(b_csr.shape[0]):
        for _ in range(n_walks):
            walk = np.empty(max_length + 1, dtype=np.int32)
            walk[0] = start
            used = 1
            current = start
            for _step in range(max_length):
                edges = b_csr.indices[b_csr.indptr[current] : b_csr.indptr[current + 1]]
                if edges.size == 0:
                    break
                probabilities = weights[edges]
                total = float(probabilities.sum())
                edge = int(rng.choice(edges, p=None if total <= 0 else probabilities / total))
                vertices = b_csc.indices[b_csc.indptr[edge] : b_csc.indptr[edge + 1]]
                current = int(rng.choice(vertices))
                walk[used] = current
                used += 1
            walks.append(walk[:used].copy())
    return walks


def _walk_pairs(walk: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    targets: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    for offset in range(1, min(window, walk.size - 1) + 1):
        targets.extend((walk[:-offset], walk[offset:]))
        contexts.extend((walk[offset:], walk[:-offset]))
    if not targets:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(targets), np.concatenate(contexts)


def _train_sgns(
    walks: list[np.ndarray],
    n_nodes: int,
    dim: int,
    window: int,
    negatives: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.ones(n_nodes, dtype=float) * 1e-12
    for walk in walks:
        counts += np.bincount(walk, minlength=n_nodes)
    negative_probability = counts**0.75
    negative_probability /= negative_probability.sum()
    scale = 0.5 / max(dim, 1)
    target_vectors = rng.uniform(-scale, scale, size=(n_nodes, dim))
    context_vectors = np.zeros((n_nodes, dim), dtype=float)

    def update(target: np.ndarray, context: np.ndarray, lr: float) -> None:
        # np.add.at may sum thousands of gradients computed from the same stale
        # vector whenever a popular node occurs repeatedly in a batch.  Clip
        # those aggregated row updates and vector norms so dense walk batches
        # (notably DBLP) cannot overflow.  This preserves the summed SGNS
        # signal, unlike replacing invalid coordinates after training.
        negative = rng.choice(
            n_nodes, size=(target.size, negatives), p=negative_probability
        )
        u = target_vectors[target].copy()
        positive_v = context_vectors[context].copy()
        negative_v = context_vectors[negative].copy()
        positive_gradient = (
            1.0 - expit(np.einsum("ij,ij->i", u, positive_v))
        ) * lr
        negative_gradient = -expit(
            np.einsum("ij,ikj->ik", u, negative_v)
        ) * lr
        target_gradient = (
            positive_gradient[:, None] * positive_v
            + np.einsum("ik,ikj->ij", negative_gradient, negative_v)
        )
        target_delta = np.zeros_like(target_vectors)
        np.add.at(target_delta, target, target_gradient)

        context_delta = np.zeros_like(context_vectors)
        np.add.at(context_delta, context, positive_gradient[:, None] * u)
        np.add.at(
            context_delta,
            negative.ravel(),
            (negative_gradient[:, :, None] * u[:, None, :]).reshape(-1, dim),
        )

        for vectors, delta in (
            (target_vectors, target_delta),
            (context_vectors, context_delta),
        ):
            delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
            delta *= np.minimum(1.0, 1.0 / np.maximum(delta_norm, 1e-12))
            vectors += delta
            vector_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors *= np.minimum(1.0, 10.0 / np.maximum(vector_norm, 1e-12))
        if not (
            np.isfinite(target_vectors).all()
            and np.isfinite(context_vectors).all()
        ):
            raise FloatingPointError(
                "SGNS produced non-finite vectors; reduce learning_rate"
            )

    order = np.arange(len(walks))
    for epoch in range(epochs):
        rng.shuffle(order)
        lr = learning_rate * max(0.1, 1.0 - epoch / max(epochs, 1))
        target_buffer: list[np.ndarray] = []
        context_buffer: list[np.ndarray] = []
        buffered = 0
        for walk_index in order:
            target, context = _walk_pairs(walks[int(walk_index)], window)
            target_buffer.append(target)
            context_buffer.append(context)
            buffered += target.size
            if buffered >= batch_size:
                target = np.concatenate(target_buffer)
                context = np.concatenate(context_buffer)
                for start in range(0, target.size, batch_size):
                    stop = start + batch_size
                    update(target[start:stop], context[start:stop], lr)
                target_buffer.clear()
                context_buffer.clear()
                buffered = 0
        if buffered:
            update(np.concatenate(target_buffer), np.concatenate(context_buffer), lr)
    norm = np.linalg.norm(target_vectors, axis=1, keepdims=True)
    embedding = target_vectors / np.maximum(norm, 1e-12)
    if not np.isfinite(embedding).all():
        raise FloatingPointError("SGNS produced a non-finite embedding")
    return embedding


def _train_ppmi_svd(
    walks: list[np.ndarray],
    n_nodes: int,
    dim: int,
    window: int,
    random_state: int | None,
) -> np.ndarray:
    """Scalable DeepWalk-style fallback when strict SGNS is too expensive."""
    cooccurrence = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    buffered = 0
    for walk in walks:
        target, context = _walk_pairs(walk, window)
        rows.append(target)
        cols.append(context)
        buffered += target.size
        if buffered >= 1_000_000:
            r = np.concatenate(rows)
            c = np.concatenate(cols)
            chunk = sparse.coo_matrix(
                (np.ones(r.size), (r, c)), shape=(n_nodes, n_nodes)
            ).tocsr()
            cooccurrence = cooccurrence + chunk
            rows.clear()
            cols.clear()
            buffered = 0
    if buffered:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        cooccurrence = cooccurrence + sparse.coo_matrix(
            (np.ones(r.size), (r, c)), shape=(n_nodes, n_nodes)
        ).tocsr()
    if cooccurrence.nnz == 0 or n_nodes == 1:
        return np.zeros((n_nodes, dim), dtype=float)
    matrix = cooccurrence.tocoo()
    row_sum = np.asarray(cooccurrence.sum(axis=1)).ravel()
    col_sum = np.asarray(cooccurrence.sum(axis=0)).ravel()
    total = float(matrix.data.sum())
    pmi = np.log(
        matrix.data * total
        / np.maximum(row_sum[matrix.row] * col_sum[matrix.col], 1e-12)
    )
    keep = pmi > 0
    ppmi = sparse.csr_matrix(
        (pmi[keep], (matrix.row[keep], matrix.col[keep])),
        shape=(n_nodes, n_nodes),
    )
    components = min(dim, n_nodes - 1)
    embedding = TruncatedSVD(
        n_components=components, random_state=random_state
    ).fit_transform(ppmi)
    if components < dim:
        embedding = np.pad(embedding, ((0, 0), (0, dim - components)))
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / np.maximum(norm, 1e-12)


def _objective_clustering(
    embedding: np.ndarray,
    b_csr: sparse.csr_matrix,
    b_csc: sparse.csc_matrix,
    edge_weights: np.ndarray,
    n_clusters: int,
    alpha: float,
    max_iter: int,
    rng: np.random.Generator,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    initial = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=random_state
    ).fit(embedding)
    labels = initial.labels_.astype(np.int32)
    centers = initial.cluster_centers_.copy()
    edge_counts = np.zeros((b_csc.shape[1], n_clusters), dtype=np.int32)
    for edge in range(b_csc.shape[1]):
        vertices = b_csc.indices[b_csc.indptr[edge] : b_csc.indptr[edge + 1]]
        edge_counts[edge] = np.bincount(labels[vertices], minlength=n_clusters)

    history: list[float] = []
    for _iteration in range(max_iter):
        cluster_sizes = np.bincount(labels, minlength=n_clusters).astype(np.int64)
        for cluster in range(n_clusters):
            members = np.flatnonzero(labels == cluster)
            if members.size:
                centers[cluster] = embedding[members].mean(axis=0)
        changes = 0
        for node in rng.permutation(embedding.shape[0]):
            old = int(labels[node])
            cost = ((centers - embedding[node]) ** 2).sum(axis=1)
            edges = b_csr.indices[b_csr.indptr[node] : b_csr.indptr[node + 1]]
            for edge in edges:
                edge_size = int(b_csc.indptr[edge + 1] - b_csc.indptr[edge])
                counts_other = edge_counts[edge].copy()
                counts_other[old] -= 1
                cost += alpha * edge_weights[edge] * (
                    (edge_size - 1) - counts_other
                )
            new = int(np.argmin(cost))
            if new != old and cluster_sizes[old] > 1:
                labels[node] = new
                cluster_sizes[old] -= 1
                cluster_sizes[new] += 1
                edge_counts[edges, old] -= 1
                edge_counts[edges, new] += 1
                changes += 1
        distance = float(((embedding - centers[labels]) ** 2).sum())
        disagreement = 0.0
        for edge in range(b_csc.shape[1]):
            q = int(edge_counts[edge].sum())
            same = int((edge_counts[edge] * (edge_counts[edge] - 1) // 2).sum())
            disagreement += edge_weights[edge] * (q * (q - 1) / 2 - same)
        history.append(distance + alpha * disagreement)
        if changes == 0:
            break
    return labels, centers, history


def _column_labels(
    b_csc: sparse.csc_matrix,
    row_labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    labels = np.empty(b_csc.shape[1], dtype=np.int32)
    for column in range(b_csc.shape[1]):
        rows = b_csc.indices[b_csc.indptr[column] : b_csc.indptr[column + 1]]
        labels[column] = int(
            np.argmax(np.bincount(row_labels[rows], minlength=n_clusters))
        )
    return labels


def _remap(labels: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    remapping: dict[object, int] = {}
    source = np.asarray(labels)
    output = np.empty(source.size, dtype=np.int32)
    for index, label in enumerate(source):
        key = label.item() if hasattr(label, "item") else label
        if key not in remapping:
            remapping[key] = len(remapping)
        output[index] = remapping[key]
    ids = np.arange(len(remapping), dtype=np.int32)
    sizes = np.bincount(output, minlength=ids.size).astype(np.int32)
    return output, ids, sizes


def labels_to_framework_state(
    incidence: Matrix | ArrayLike,
    row_labels: ArrayLike,
    column_labels: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, sparse.csr_matrix]:
    """Convert labels to the exact contract consumed by summary_utils.

    Returns ``rcids, ccids, Nx, Ny, Qx, Qy, DnZ``.  ``DnZ`` counts nonzero
    incidences in every row-cluster/column-cluster block.
    """
    a = _as_binary_csr(incidence)
    row = np.asarray(row_labels)
    column = np.asarray(column_labels)
    if row.ndim != 1 or row.size != a.shape[0]:
        raise ValueError("row_labels length must equal the number of rows")
    if column.ndim != 1 or column.size != a.shape[1]:
        raise ValueError("column_labels length must equal the number of columns")
    qx, rcids, nx = _remap(row)
    qy, ccids, ny = _remap(column)
    row_membership = sparse.csr_matrix(
        (np.ones(a.shape[0]), (np.arange(a.shape[0]), qx)),
        shape=(a.shape[0], nx.size),
    )
    column_membership = sparse.csr_matrix(
        (np.ones(a.shape[1]), (np.arange(a.shape[1]), qy)),
        shape=(a.shape[1], ny.size),
    )
    dnz = (row_membership.T @ a @ column_membership).tocsr()
    return rcids, ccids, nx, ny, qx, qy, dnz


@dataclass
class GAHGC:
    n_clusters: int
    alpha: float = 0.5
    max_walk_length: int = 50
    n_walks: int = 50
    embedding_dim: int = 32
    window: int = 5
    negative_samples: int = 5
    embedding_epochs: int = 1
    learning_rate: float = 0.025
    batch_size: int = 4096
    max_iter: int = 100
    bandwidth: float | Literal["median"] = "median"
    embedding_backend: Literal["sgns", "ppmi_svd"] = "sgns"
    random_state: int | None = 0

    row_labels_: np.ndarray = field(init=False, repr=False)
    column_labels_: np.ndarray = field(init=False, repr=False)
    embedding_: np.ndarray = field(init=False, repr=False)
    incidence_: sparse.csr_matrix = field(init=False, repr=False)
    hyperedge_weights_: np.ndarray = field(init=False, repr=False)
    centers_: np.ndarray = field(init=False, repr=False)
    objective_history_: list[float] = field(init=False, repr=False)
    bandwidth_: float = field(init=False, default=1.0)

    def fit(self, x: Matrix | ArrayLike) -> "GAHGC":
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if min(
            self.max_walk_length,
            self.n_walks,
            self.embedding_dim,
            self.window,
            self.negative_samples,
            self.embedding_epochs,
            self.batch_size,
            self.max_iter,
        ) < 1:
            raise ValueError("GAHGC iteration and embedding settings must be positive")
        incidence = _as_binary_csr(x)
        if self.n_clusters > incidence.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of nodes")
        if np.any(np.diff(incidence.indptr) == 0):
            warnings.warn(
                "isolated rows have random length-one-walk embeddings",
                RuntimeWarning,
            )
        incidence_csc = incidence.tocsc()
        weights, bandwidth = _hyperedge_weights(
            incidence, incidence_csc, self.bandwidth
        )
        rng = np.random.default_rng(self.random_state)
        walks = _sample_walks(
            incidence,
            incidence_csc,
            weights,
            self.n_walks,
            self.max_walk_length,
            rng,
        )
        if self.embedding_backend == "sgns":
            embedding = _train_sgns(
                walks,
                incidence.shape[0],
                self.embedding_dim,
                self.window,
                self.negative_samples,
                self.embedding_epochs,
                self.learning_rate,
                self.batch_size,
                rng,
            )
        elif self.embedding_backend == "ppmi_svd":
            embedding = _train_ppmi_svd(
                walks,
                incidence.shape[0],
                self.embedding_dim,
                self.window,
                self.random_state,
            )
        else:
            raise ValueError("embedding_backend must be 'sgns' or 'ppmi_svd'")
        row_labels, centers, history = _objective_clustering(
            embedding,
            incidence,
            incidence_csc,
            weights,
            self.n_clusters,
            self.alpha,
            self.max_iter,
            rng,
            self.random_state,
        )
        self.row_labels_ = row_labels
        self.column_labels_ = _column_labels(
            incidence_csc, row_labels, self.n_clusters
        )
        self.embedding_ = embedding
        self.incidence_ = incidence
        self.hyperedge_weights_ = weights
        self.centers_ = centers
        self.objective_history_ = history
        self.bandwidth_ = bandwidth
        return self

    def fit_predict(self, x: Matrix | ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        self.fit(x)
        return self.row_labels_.copy(), self.column_labels_.copy()
