"""Controlled practical-scalability experiment for HyDRA.

For each dataset and random seed, the script creates nested samples of whole
hyperedges.  Isolated rows are removed after sampling.  It then runs HyDRA
with fixed parameters and checkpoints runtime, merge rounds, summary cost,
and final cluster counts after every run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

import hyp_sum as hs


RUN_COLUMNS = [
    "Dataset",
    "Seed",
    "Fraction",
    "Nodes",
    "Hyperedges",
    "Nonzeros",
    "Density",
    "Runtime (s)",
    "Merge Rounds",
    "Adaptive-LSH Rounds",
    "Final-LSH Rounds",
    "Final Node Clusters",
    "Final Hyperedge Clusters",
    "Original Cost",
    "Summary Cost",
    "Cost Improvement",
    "Status",
    "Error",
]


def _is_true(value) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _load_config(defaults_path: str, config_path: str) -> dict:
    with open(defaults_path, encoding="utf-8") as stream:
        config = json.load(stream)
    with open(config_path, encoding="utf-8") as stream:
        config.update(json.load(stream))
    return config


def _data_path(data_dir: str, dataset: str) -> str:
    if dataset.endswith(".csv"):
        return os.path.join(data_dir, dataset)
    return os.path.join(data_dir, dataset, "A.mat")


def nested_hyperedge_sample(
        incidence,
        fraction: float,
        column_order: np.ndarray) -> sparse.csr_matrix:
    """Return a nested whole-hyperedge sample with isolated nodes removed."""
    if not 0 < fraction <= 1:
        raise ValueError("fractions must lie in (0, 1]")
    incidence = sparse.csr_matrix(incidence)
    if column_order.shape != (incidence.shape[1],):
        raise ValueError("column_order must contain one entry per hyperedge")
    number = max(1, min(incidence.shape[1], math.ceil(fraction * incidence.shape[1])))
    columns = np.sort(column_order[:number])
    sampled = incidence[:, columns].tocsr()
    nonempty_rows = np.asarray(sampled.getnnz(axis=1)).ravel() > 0
    sampled = sampled[nonempty_rows].tocsr()
    sampled.eliminate_zeros()
    return sampled


def _iteration_counts(stats: Iterable) -> tuple[int, int, int]:
    procedures = [str(row[0]) for row in stats]
    adaptive = sum(name == "COS-COL-rb" for name in procedures)
    final = sum(name == "COS-COL" for name in procedures)
    return adaptive + final, adaptive, final


def _write_checkpoint(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.DataFrame(rows, columns=RUN_COLUMNS).to_csv(temporary, index=False)
    os.replace(temporary, output_path)


def run_scalability_experiment(
        defaults_path: str,
        config_path: str,
        datasets: list[str],
        fractions: list[float],
        seeds: list[int],
        output_path: str,
        fresh: bool = False) -> pd.DataFrame:
    """Run or resume the HyDRA scaling experiment."""
    # The data loader imports optional experiment dependencies; keep it lazy so
    # command-line help and the plotting function work in lightweight setups.
    import utils as ut

    config = _load_config(defaults_path, config_path)
    output = Path(output_path)
    if fresh and output.exists():
        output.unlink()

    rows = []
    if output.exists():
        rows = pd.read_csv(output).to_dict("records")
    completed = {
        (str(row["Dataset"]), int(row["Seed"]), float(row["Fraction"]))
        for row in rows
        if row.get("Status") == "ok"
    }

    loaded = {}
    for dataset in datasets:
        if dataset not in config["datasets"]:
            raise KeyError(f"dataset {dataset!r} is absent from the configuration")
        weighted = _is_true(config["datasets"][dataset].get("weighted", False))
        path = _data_path(config["data_dir"], dataset)
        incidence, _, _, _ = ut.load_matrix_and_hypergraph(path, weighted=weighted)
        loaded[dataset] = sparse.csr_matrix(incidence)

    fractions = sorted(set(float(value) for value in fractions))
    if not fractions or fractions[0] <= 0 or fractions[-1] > 1:
        raise ValueError("fractions must lie in (0, 1]")

    total = len(datasets) * len(seeds) * len(fractions)
    done = len(completed)
    for dataset in datasets:
        incidence = loaded[dataset]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            column_order = rng.permutation(incidence.shape[1])
            for fraction in fractions:
                key = (dataset, seed, fraction)
                if key in completed:
                    continue
                done += 1
                print(
                    f"[{done}/{total}] dataset={dataset} seed={seed} "
                    f"fraction={fraction:g}",
                    flush=True,
                )
                sampled = nested_hyperedge_sample(
                    incidence, fraction, column_order
                )
                row = {
                    "Dataset": dataset,
                    "Seed": seed,
                    "Fraction": fraction,
                    "Nodes": sampled.shape[0],
                    "Hyperedges": sampled.shape[1],
                    "Nonzeros": sampled.nnz,
                    "Density": sampled.nnz / (sampled.shape[0] * sampled.shape[1]),
                    "Runtime (s)": np.nan,
                    "Merge Rounds": np.nan,
                    "Adaptive-LSH Rounds": np.nan,
                    "Final-LSH Rounds": np.nan,
                    "Final Node Clusters": np.nan,
                    "Final Hyperedge Clusters": np.nan,
                    "Original Cost": hs.init_cost_hs(sampled),
                    "Summary Cost": np.nan,
                    "Cost Improvement": np.nan,
                    "Status": "error",
                    "Error": "",
                }
                started = time.perf_counter()
                try:
                    result = hs.hc_search_hs(
                        sampled,
                        seed=seed,
                        r=int(config["r"]),
                        b=int(config["b"]),
                        min_size=int(config["min_size"]),
                        max_trials=int(config["max_trials"]),
                        max_no_improvements=int(config["max_no_improvements"]),
                        cost_threshold=float(config["algorithms"]["HyDRA"]),
                        weights=config["weights"],
                        verbose=False,
                    )
                    runtime = time.perf_counter() - started
                    final_node_clusters = int(result[0])
                    final_hyperedge_clusters = int(result[1])
                    summary_cost = float(result[7])
                    iterations, adaptive, final = _iteration_counts(result[9])
                    row.update({
                        "Runtime (s)": runtime,
                        "Merge Rounds": iterations,
                        "Adaptive-LSH Rounds": adaptive,
                        "Final-LSH Rounds": final,
                        "Final Node Clusters": final_node_clusters,
                        "Final Hyperedge Clusters": final_hyperedge_clusters,
                        "Summary Cost": summary_cost,
                        "Cost Improvement": 1 - summary_cost / row["Original Cost"],
                        "Status": "ok",
                    })
                except Exception as error:
                    row["Runtime (s)"] = time.perf_counter() - started
                    row["Error"] = f"{type(error).__name__}: {error}"
                    print(f"  failed: {row['Error']}", flush=True)
                rows.append(row)
                _write_checkpoint(rows, output)

    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def plot_scalability_results(results, save_path=None):
    """Plot runtime, merge rounds, and quality against sampled input size."""
    data = pd.read_csv(results) if isinstance(results, (str, os.PathLike)) else results.copy()
    data = data[data["Status"] == "ok"].copy()
    if data.empty:
        raise ValueError("no successful scalability runs are available")

    grouped = data.groupby(["Dataset", "Fraction"], as_index=False).agg(
        Nodes=("Nodes", "mean"),
        Hyperedges=("Hyperedges", "mean"),
        Nonzeros=("Nonzeros", "mean"),
        Runtime=("Runtime (s)", "mean"),
        Runtime_SD=("Runtime (s)", "std"),
        Merge_Rounds=("Merge Rounds", "mean"),
        Merge_Rounds_SD=("Merge Rounds", "std"),
        Cost_Improvement=("Cost Improvement", "mean"),
        Cost_Improvement_SD=("Cost Improvement", "std"),
        Runs=("Seed", "nunique"),
    ).fillna(0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for dataset, subset in grouped.groupby("Dataset", sort=False):
        subset = subset.sort_values("Nonzeros")
        axes[0].errorbar(
            subset["Nonzeros"], subset["Runtime"],
            yerr=subset["Runtime_SD"], marker="o", capsize=3,
            label=dataset,
        )
        axes[1].errorbar(
            subset["Nonzeros"], subset["Merge_Rounds"],
            yerr=subset["Merge_Rounds_SD"], marker="o", capsize=3,
            label=dataset,
        )
        axes[2].errorbar(
            subset["Nonzeros"], subset["Cost_Improvement"],
            yerr=subset["Cost_Improvement_SD"], marker="o", capsize=3,
            label=dataset,
        )

    axes[0].set_yscale("log")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel(r"Incidence entries $\mathrm{nnz}(A)$")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Runtime (s, log scale)")
    axes[1].set_ylabel("Merge rounds")
    axes[2].set_ylabel("Cost improvement")
    axes[0].legend(title="Dataset", frameon=False)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return grouped, fig, axes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defaults", default="config/defaults.json")
    parser.add_argument("--config", default="config/summ.json")
    parser.add_argument(
        "--datasets", nargs="+", default=["classic", "house"],
        help="use at least one weak- and one strong-structure dataset",
    )
    parser.add_argument(
        "--fractions", nargs="+", type=float,
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output", default="../out/hydra_scalability.csv",
    )
    parser.add_argument(
        "--plot", default=None,
        help="optional output path for a PDF/PNG scalability figure",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="discard an existing checkpoint instead of resuming it",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_scalability_experiment(
        args.defaults,
        args.config,
        args.datasets,
        args.fractions,
        args.seeds,
        args.output,
        fresh=args.fresh,
    )
    if args.plot:
        plot_scalability_results(results, args.plot)


if __name__ == "__main__":
    main()
