"""Independent implementation of DHCC for the HyDRA framework.

This module implements Algorithm 1 of Song et al., *Manifold-aware dual
hypergraph co-clustering* (Neurocomputing, 2026): k-nearest-neighbour
hypergraphs are constructed for the rows and columns, their unnormalised
hypergraph Laplacians regularise a semi-nonnegative matrix
tri-factorisation, and the row/column partitions are read from the two
nonnegative factors.

This is a paper-based reimplementation, not official author code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


Matrix = np.ndarray | sparse.spmatrix


def _as_float_csr(x: Matrix | ArrayLike) -> sparse.csr_matrix:
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
    return out


def _positive_negative(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(x)
    return 0.5 * (absolute + x), 0.5 * (absolute - x)


def _knn_hypergraph(
    observations: sparse.csr_matrix,
    k: int,
    metric: str,
    n_jobs: int | None,
) -> tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix]:
    """Construct H, hyperedge weights (Eq. 2), and HWD_e^-1H^T.

    Every observation is the reference vertex of one hyperedge containing
    itself and its k nearest neighbours.  The paper's K in Equation 2 is the
    resulting hyperedge cardinality, including the reference vertex.
    """
    n_vertices = observations.shape[0]
    neighbours_per_edge = min(k + 1, n_vertices)
    search = NearestNeighbors(
        n_neighbors=neighbours_per_edge,
        metric=metric,
        algorithm="brute",
        n_jobs=n_jobs,
    ).fit(observations)
    distances, neighbours = search.kneighbors(observations, return_distance=True)

    squared = distances * distances
    scale = squared.mean(axis=1)
    ratios = np.divide(
        squared,
        scale[:, None],
        out=np.zeros_like(squared),
        where=scale[:, None] > 0,
    )
    edge_weights = np.exp(-ratios).sum(axis=1)

    rows = neighbours.ravel()
    columns = np.repeat(np.arange(n_vertices), neighbours_per_edge)
    incidence = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, columns)),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    edge_degrees = np.asarray(incidence.sum(axis=0)).ravel()
    scaled = incidence.multiply(
        np.divide(
            edge_weights,
            edge_degrees,
            out=np.zeros_like(edge_weights),
            where=edge_degrees > 0,
        )
    )
    affinity = (scaled @ incidence.T).tocsr()
    vertex_degrees = np.asarray(incidence @ edge_weights).ravel()
    return affinity, vertex_degrees, incidence


def _partition_initialisation(
    observations: sparse.csr_matrix,
    n_clusters: int,
    random_state: int | None,
    n_init: int,
    max_iter: int,
    floor: float,
) -> np.ndarray:
    labels = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    ).fit_predict(observations)
    factor = np.full((observations.shape[0], n_clusters), floor, dtype=np.float64)
    factor[np.arange(observations.shape[0]), labels] = 1.0
    return _normalise_columns(factor, floor)


def _normalise_columns(factor: np.ndarray, floor: float) -> np.ndarray:
    factor = np.maximum(factor, floor)
    norms = np.linalg.norm(factor, axis=0, keepdims=True)
    return factor / np.maximum(norms, floor)


def _core_update(
    x: sparse.csr_matrix,
    u: np.ndarray,
    v: np.ndarray,
    ridge: float,
) -> np.ndarray:
    left = u.T @ u + ridge * np.eye(u.shape[1])
    right = v.T @ v + ridge * np.eye(v.shape[1])
    middle = u.T @ (x @ v)
    # Equivalent to (U^T U)^-1 U^T X V (V^T V)^-1 (Eq. 10),
    # with a tiny ridge for singular/near-singular partition matrices.
    left_solved = np.linalg.solve(left, middle)
    return np.linalg.solve(right.T, left_solved.T).T


def _objective(
    x_squared_norm: float,
    u: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
    sample_affinity: sparse.csr_matrix,
    sample_degree: np.ndarray,
    feature_affinity: sparse.csr_matrix,
    feature_degree: np.ndarray,
    sample_regularisation: float,
    feature_regularisation: float,
    x: sparse.csr_matrix,
) -> float:
    utu = u.T @ u
    vtv = v.T @ v
    reconstruction = (
        x_squared_norm
        - 2.0 * float(np.sum((u @ s) * (x @ v)))
        + float(np.trace(utu @ s @ vtv @ s.T))
    )
    sample_smoothness = float(
        np.sum(sample_degree[:, None] * u * u)
        - np.sum(u * (sample_affinity @ u))
    )
    feature_smoothness = float(
        np.sum(feature_degree[:, None] * v * v)
        - np.sum(v * (feature_affinity @ v))
    )
    return (
        reconstruction
        + sample_regularisation * sample_smoothness
        + feature_regularisation * feature_smoothness
    )


@dataclass
class DHCC:
    """Dual Hypergraph Co-Clustering estimator following paper Algorithm 1."""

    n_row_clusters: int
    n_column_clusters: int
    sample_regularisation: float = 100.0
    feature_regularisation: float = 100.0
    n_neighbors: int = 3
    max_iter: int = 100
    tol: float = 1e-5
    metric: Literal["euclidean", "cosine"] = "euclidean"
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    ridge: float = 1e-8
    floor: float = 1e-12
    n_jobs: int | None = None
    random_state: int | None = 0

    row_labels_: np.ndarray = field(init=False, repr=False)
    column_labels_: np.ndarray = field(init=False, repr=False)
    row_factor_: np.ndarray = field(init=False, repr=False)
    column_factor_: np.ndarray = field(init=False, repr=False)
    core_: np.ndarray = field(init=False, repr=False)
    objective_history_: list[float] = field(init=False, repr=False)
    sample_hypergraph_: sparse.csr_matrix = field(init=False, repr=False)
    feature_hypergraph_: sparse.csr_matrix = field(init=False, repr=False)

    def fit(self, x: Matrix | ArrayLike) -> "DHCC":
        x_csr = _as_float_csr(x)
        n_rows, n_columns = x_csr.shape
        if not 1 <= self.n_row_clusters <= n_rows:
            raise ValueError("n_row_clusters must be between 1 and X.shape[0]")
        if not 1 <= self.n_column_clusters <= n_columns:
            raise ValueError("n_column_clusters must be between 1 and X.shape[1]")
        if self.sample_regularisation < 0 or self.feature_regularisation < 0:
            raise ValueError("regularisation parameters must be nonnegative")
        if self.n_neighbors < 1 or self.max_iter < 1:
            raise ValueError("n_neighbors and max_iter must be positive")
        if self.tol < 0 or self.ridge <= 0 or self.floor <= 0:
            raise ValueError("tol must be nonnegative; ridge and floor must be positive")

        sample_affinity, sample_degree, sample_h = _knn_hypergraph(
            x_csr, self.n_neighbors, self.metric, self.n_jobs
        )
        feature_data = x_csr.T.tocsr()
        feature_affinity, feature_degree, feature_h = _knn_hypergraph(
            feature_data, self.n_neighbors, self.metric, self.n_jobs
        )
        u = _partition_initialisation(
            x_csr,
            self.n_row_clusters,
            self.random_state,
            self.kmeans_n_init,
            self.kmeans_max_iter,
            self.floor,
        )
        v = _partition_initialisation(
            feature_data,
            self.n_column_clusters,
            self.random_state,
            self.kmeans_n_init,
            self.kmeans_max_iter,
            self.floor,
        )

        history: list[float] = []
        x_squared_norm = float(x_csr.multiply(x_csr).sum())
        for _ in range(self.max_iter):
            s = _core_update(x_csr, u, v, self.ridge)

            a = np.asarray(x_csr @ (v @ s.T))
            b = s @ (v.T @ v) @ s.T
            a_positive, a_negative = _positive_negative(a)
            b_positive, b_negative = _positive_negative(b)
            numerator = (
                a_positive
                + u @ b_negative
                + self.sample_regularisation * (sample_affinity @ u)
            )
            denominator = (
                a_negative
                + u @ b_positive
                + self.sample_regularisation * sample_degree[:, None] * u
            )
            u *= np.sqrt(
                np.maximum(numerator, self.floor)
                / np.maximum(denominator, self.floor)
            )
            u = _normalise_columns(u, self.floor)

            p = np.asarray(x_csr.T @ (u @ s))
            q = s.T @ (u.T @ u) @ s
            p_positive, p_negative = _positive_negative(p)
            q_positive, q_negative = _positive_negative(q)
            numerator = (
                p_positive
                + v @ q_negative
                + self.feature_regularisation * (feature_affinity @ v)
            )
            denominator = (
                p_negative
                + v @ q_positive
                + self.feature_regularisation * feature_degree[:, None] * v
            )
            v *= np.sqrt(
                np.maximum(numerator, self.floor)
                / np.maximum(denominator, self.floor)
            )
            v = _normalise_columns(v, self.floor)

            s_for_objective = _core_update(x_csr, u, v, self.ridge)
            value = _objective(
                x_squared_norm,
                u,
                s_for_objective,
                v,
                sample_affinity,
                sample_degree,
                feature_affinity,
                feature_degree,
                self.sample_regularisation,
                self.feature_regularisation,
                x_csr,
            )
            history.append(value)
            if len(history) > 1:
                change = abs(history[-2] - history[-1])
                if change <= self.tol * max(1.0, abs(history[-2])):
                    break

        self.row_factor_ = u
        self.column_factor_ = v
        self.core_ = _core_update(x_csr, u, v, self.ridge)
        self.row_labels_ = np.argmax(u, axis=1).astype(np.int32)
        self.column_labels_ = np.argmax(v, axis=1).astype(np.int32)
        self.objective_history_ = history
        self.sample_hypergraph_ = sample_h
        self.feature_hypergraph_ = feature_h
        return self

    def fit_predict(self, x: Matrix | ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        self.fit(x)
        return self.row_labels_.copy(), self.column_labels_.copy()
