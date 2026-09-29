"""Select global GAHGC/DHCC settings using HyDRA's storage objective.

The tuner evaluates each candidate on user-designated development datasets and
minimizes the dataset-balanced mean of ``Cost / Orig. Cost``.  It never uses
class labels.  The selected configuration is global: one setting per algorithm
is written to ``summ_tuned.json`` and can then be frozen for the remaining
datasets.

Example
-------
python code/tune_reimplementations.py \
    --defaults config/defaults.json \
    --config config/summ.json \
    --datasets senate dblp \
    --output-dir ../out/reimplementation_tuning
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import numpy as np

from dhcc import DHCC
from gahgc import GAHGC, labels_to_framework_state


JsonDict = dict[str, Any]


def _deep_merge(base: JsonDict, override: JsonDict) -> JsonDict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _unique(values: Iterable[JsonDict]) -> list[JsonDict]:
    found: set[str] = set()
    output: list[JsonDict] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if key not in found:
            found.add(key)
            output.append(value)
    return output


def compact_candidates(algorithm: str) -> list[JsonDict]:
    """A computationally reasonable, paper-guided sensitivity search."""
    if algorithm == "gahgc":
        base = {
            "alpha": 0.5,
            "max_walk_length": 50,
            "n_walks": 50,
            # The paper defines delta but does not report a value.  A median
            # scale is invariant to dataset units and uses no labels.
            "bandwidth": "median",
        }
        candidates = [base.copy()]
        for key, values in {
            "alpha": [0.3, 0.5, 0.7],
            "max_walk_length": [50, 100, 150],
            "n_walks": [50, 100, 200],
        }.items():
            for value in values:
                candidate = base.copy()
                candidate[key] = value
                candidates.append(candidate)
        return _unique(candidates)

    if algorithm == "dhcc":
        # The paper's cross-dataset choice is k=3 and lambda=mu=100.  The
        # compact search varies k and a shared regularization strength, which
        # reduces 64 combinations to 16 while retaining the dual penalty.
        return [
            {
                "n_neighbors": k,
                "sample_regularisation": regularisation,
                "feature_regularisation": regularisation,
            }
            for k, regularisation in itertools.product(
                [2, 3, 5, 10], [1.0, 10.0, 100.0, 1000.0]
            )
        ]
    raise ValueError(f"unsupported algorithm: {algorithm}")


def paper_candidates(algorithm: str) -> list[JsonDict]:
    """Larger grids based on the papers; potentially very expensive."""
    if algorithm == "gahgc":
        return [
            {
                "alpha": alpha,
                "max_walk_length": walk_length,
                "n_walks": n_walks,
                "bandwidth": "median",
            }
            for alpha, walk_length, n_walks in itertools.product(
                [0.3, 0.5, 0.7],
                [50, 100, 150],
                [50, 100, 200],
            )
        ]
    if algorithm == "dhcc":
        regularisation = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
        return [
            {
                "n_neighbors": k,
                "sample_regularisation": sample,
                "feature_regularisation": feature,
            }
            for k, sample, feature in itertools.product(
                range(1, 11), regularisation, regularisation
            )
        ]
    raise ValueError(f"unsupported algorithm: {algorithm}")


def candidate_grid(algorithm: str, search: str) -> list[JsonDict]:
    if search == "compact":
        return compact_candidates(algorithm)
    if search == "paper":
        return paper_candidates(algorithm)
    raise ValueError("search must be 'compact' or 'paper'")


def _dataset_path(data_dir: str, dataset: str) -> str:
    if dataset.endswith(".csv"):
        return os.path.join(data_dir, dataset)
    return os.path.join(data_dir, dataset, "A.mat")


def _is_true(value: Any) -> bool:
    return value == "True" if isinstance(value, str) else bool(value)


def _write_original_summary(
    summary_dir: str,
    dataset: str,
    a,
    hyperedges: list[list[int]],
    node_id_map: dict,
    hedge_weights: list,
) -> None:
    import summary_utils as sut

    vertices = {index: [index] for index in range(a.shape[0])}
    weights = hedge_weights if len(hedge_weights) > 0 else [1] * len(hyperedges)
    inverse_map = {value: key for key, value in node_id_map.items()}
    path = os.path.join(summary_dir, f"original__data={dataset}.csv")
    sut.save_summary(
        vertices,
        hyperedges,
        weights,
        [],
        path,
        inv_node_id_map=inverse_map,
    )


def _fit_model(
    algorithm: str,
    a,
    dataset: str,
    config: JsonDict,
    candidate: JsonDict,
    seed: int,
):
    parameters = _deep_merge(config["algorithms"][algorithm], candidate)
    use_means = _is_true(config.get("use_means", True))
    cluster_pair = config["datasets"][dataset]["means_nclusters"]

    if algorithm == "gahgc":
        n_clusters = (
            cluster_pair[0]
            if use_means
            else parameters.get("default_nclusters", 9)
        )
        model = GAHGC(
            n_clusters=n_clusters,
            alpha=parameters.get("alpha", 0.5),
            max_walk_length=parameters.get("max_walk_length", 50),
            n_walks=parameters.get("n_walks", 50),
            embedding_dim=parameters.get("embedding_dim", 32),
            window=parameters.get("window", 5),
            negative_samples=parameters.get("negative_samples", 5),
            embedding_epochs=parameters.get("embedding_epochs", 1),
            learning_rate=parameters.get("learning_rate", 0.025),
            batch_size=parameters.get("batch_size", 4096),
            max_iter=parameters.get("max_iter", 100),
            bandwidth=parameters.get("bandwidth", "median"),
            embedding_backend=parameters.get("embedding_backend", "sgns"),
            random_state=seed,
        ).fit(a)
        return model, n_clusters, -1, parameters

    if algorithm == "dhcc":
        if use_means:
            row_clusters, column_clusters = cluster_pair
        else:
            row_clusters = parameters.get("default_nclusters", 9)
            column_clusters = parameters.get(
                "default_n_column_clusters", row_clusters
            )
        model = DHCC(
            n_row_clusters=row_clusters,
            n_column_clusters=column_clusters,
            sample_regularisation=parameters.get(
                "sample_regularisation", 100.0
            ),
            feature_regularisation=parameters.get(
                "feature_regularisation", 100.0
            ),
            n_neighbors=parameters.get("n_neighbors", 3),
            max_iter=parameters.get("max_iter", 100),
            tol=parameters.get("tol", 1e-5),
            metric=parameters.get("metric", "euclidean"),
            kmeans_n_init=parameters.get("kmeans_n_init", 10),
            kmeans_max_iter=parameters.get("kmeans_max_iter", 300),
            ridge=parameters.get("ridge", 1e-8),
            floor=parameters.get("floor", 1e-12),
            n_jobs=parameters.get("n_jobs", None),
            random_state=seed,
        ).fit(a)
        return model, row_clusters, column_clusters, parameters

    raise ValueError(f"unsupported algorithm: {algorithm}")


RUN_FIELDS = [
    "algorithm",
    "candidate_id",
    "dataset",
    "seed",
    "parameters",
    "cost",
    "original_cost",
    "relative_cost",
    "time_seconds",
    "node_clusters",
    "hyperedge_clusters",
    "status",
    "error",
]


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_runs(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_tuning(
    config: JsonDict,
    datasets: list[str],
    algorithms: list[str],
    seeds: list[int],
    output_dir: str,
    search: str = "compact",
    restart: bool = False,
    keep_summaries: bool = False,
    candidates_override: dict[str, list[JsonDict]] | None = None,
) -> JsonDict:
    import summary_utils as sut
    import utils as ut

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_dir = output / "summaries"
    summary_dir.mkdir(exist_ok=True)
    runs_path = output / "tuning_runs.csv"

    if restart:
        previous_rows: list[JsonDict] = []
    else:
        previous_rows = _read_runs(runs_path)
    completed = {
        (
            row["algorithm"],
            row["candidate_id"],
            row["dataset"],
            int(row["seed"]),
            row["parameters"],
        )
        for row in previous_rows
        if row.get("status") == "ok"
    }
    rows = previous_rows

    loaded: dict[str, tuple] = {}
    for dataset in datasets:
        if dataset not in config["datasets"]:
            raise KeyError(f"dataset {dataset!r} is missing from the configuration")
        path = _dataset_path(config["data_dir"], dataset)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        weighted = _is_true(config["datasets"][dataset]["weighted"])
        a, hyperedges, node_id_map, hedge_weights = \
            ut.load_matrix_and_hypergraph(path, weighted=weighted)
        _write_original_summary(
            str(summary_dir),
            dataset,
            a,
            hyperedges,
            node_id_map,
            hedge_weights,
        )
        loaded[dataset] = (a, hyperedges, node_id_map, hedge_weights)

    candidate_map: dict[str, list[JsonDict]] = {}
    for algorithm in algorithms:
        candidate_map[algorithm] = (
            candidates_override[algorithm]
            if candidates_override is not None
            else candidate_grid(algorithm, search)
        )

    total = sum(len(candidate_map[a]) for a in algorithms) * len(datasets) * len(seeds)
    planned = {
        (
            algorithm,
            f"{algorithm}-{candidate_number:03d}",
            dataset,
            seed,
            json.dumps(candidate, sort_keys=True),
        )
        for algorithm in algorithms
        for candidate_number, candidate in enumerate(candidate_map[algorithm])
        for dataset in datasets
        for seed in seeds
    }
    done = len(completed & planned)
    for algorithm in algorithms:
        for candidate_number, candidate in enumerate(candidate_map[algorithm]):
            candidate_id = f"{algorithm}-{candidate_number:03d}"
            candidate_json = json.dumps(candidate, sort_keys=True)
            for dataset in datasets:
                a, _, node_id_map, hedge_weights = loaded[dataset]
                for seed in seeds:
                    key = (algorithm, candidate_id, dataset, seed, candidate_json)
                    if key in completed:
                        continue
                    done += 1
                    print(
                        f"[{done}/{total}] {algorithm} {candidate_id} "
                        f"dataset={dataset} seed={seed}",
                        flush=True,
                    )
                    started = time.perf_counter()
                    row: JsonDict = {
                        "algorithm": algorithm,
                        "candidate_id": candidate_id,
                        "dataset": dataset,
                        "seed": seed,
                        "parameters": candidate_json,
                        "cost": "",
                        "original_cost": "",
                        "relative_cost": "",
                        "time_seconds": "",
                        "node_clusters": "",
                        "hyperedge_clusters": "",
                        "status": "error",
                        "error": "",
                    }
                    try:
                        model, row_clusters, column_clusters, _ = _fit_model(
                            algorithm, a, dataset, config, candidate, seed
                        )
                        rcids, ccids, nx, ny, qx, qy, dnz = \
                            labels_to_framework_state(
                                a, model.row_labels_, model.column_labels_
                            )
                        tuning_name = f"tuning-{candidate_id}"
                        result = sut.get_and_evaluate_summary(
                            nx,
                            ny,
                            qx,
                            qy,
                            dnz,
                            ccids,
                            rcids,
                            a,
                            dataset,
                            tuning_name,
                            -1,
                            row_clusters,
                            column_clusters,
                            -1,
                            -1,
                            -1,
                            seed,
                            str(summary_dir),
                            node_id_map=node_id_map,
                            hedge_weights=hedge_weights,
                        )
                        cost = float(result["Cost"])
                        original_cost = float(result["Orig. Cost"])
                        row.update(
                            {
                                "cost": cost,
                                "original_cost": original_cost,
                                "relative_cost": cost / original_cost,
                                "time_seconds": time.perf_counter() - started,
                                "node_clusters": result["Num Node Clusters"],
                                "hyperedge_clusters": result[
                                    "Num Hedge Clusters"
                                ],
                                "status": "ok",
                            }
                        )
                        if not keep_summaries:
                            generated = sut.get_summary_name(
                                dataset,
                                tuning_name,
                                r=row_clusters,
                                b=column_clusters,
                                run=seed,
                            )
                            generated_path = summary_dir / generated
                            if generated_path.exists():
                                generated_path.unlink()
                    except Exception as error:  # keep the search resumable
                        row["time_seconds"] = time.perf_counter() - started
                        row["error"] = f"{type(error).__name__}: {error}"
                        print(f"  failed: {row['error']}", flush=True)
                    rows.append(row)
                    _write_csv(runs_path, rows, RUN_FIELDS)

    expected_runs = len(datasets) * len(seeds)
    grouped: dict[tuple[str, str, str], list[JsonDict]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[
                (row["algorithm"], row["candidate_id"], row["parameters"])
            ].append(row)

    candidate_rows: list[JsonDict] = []
    selected: JsonDict = {}
    for algorithm in algorithms:
        eligible: list[JsonDict] = []
        for candidate_number, candidate in enumerate(candidate_map[algorithm]):
            candidate_id = f"{algorithm}-{candidate_number:03d}"
            candidate_json = json.dumps(candidate, sort_keys=True)
            values = grouped.get((algorithm, candidate_id, candidate_json), [])
            values = [
                row for row in values
                if row["dataset"] in datasets and int(row["seed"]) in seeds
            ]
            unique_runs = {
                (row["dataset"], int(row["seed"])) for row in values
            }
            if len(unique_runs) != expected_runs:
                continue
            by_dataset: dict[str, list[float]] = defaultdict(list)
            runtimes: list[float] = []
            for row in values:
                by_dataset[row["dataset"]].append(float(row["relative_cost"]))
                runtimes.append(float(row["time_seconds"]))
            dataset_scores = [mean(by_dataset[name]) for name in datasets]
            summary = {
                "algorithm": algorithm,
                "candidate_id": candidate_id,
                "parameters": json.dumps(candidate, sort_keys=True),
                "mean_relative_cost": mean(dataset_scores),
                "std_across_datasets": (
                    pstdev(dataset_scores) if len(dataset_scores) > 1 else 0.0
                ),
                "mean_time_seconds": mean(runtimes),
                "completed_runs": len(unique_runs),
            }
            candidate_rows.append(summary)
            eligible.append(summary)
        if not eligible:
            raise RuntimeError(
                f"no complete candidate is available for {algorithm}; "
                f"inspect {runs_path} and rerun to resume"
            )
        best = min(
            eligible,
            key=lambda item: (
                item["mean_relative_cost"],
                item["mean_time_seconds"],
                item["candidate_id"],
            ),
        )
        selected[algorithm] = {
            "candidate_id": best["candidate_id"],
            "parameters": json.loads(best["parameters"]),
            "mean_relative_cost": best["mean_relative_cost"],
            "std_across_datasets": best["std_across_datasets"],
        }

    _write_csv(
        output / "tuning_candidates.csv",
        candidate_rows,
        [
            "algorithm",
            "candidate_id",
            "parameters",
            "mean_relative_cost",
            "std_across_datasets",
            "mean_time_seconds",
            "completed_runs",
        ],
    )
    selection = {
        "selection_objective": (
            "mean across development datasets of the seed-mean "
            "summary Cost / Orig. Cost"
        ),
        "development_datasets": datasets,
        "seeds": seeds,
        "search": search,
        "selected": selected,
    }
    with (output / "selected_settings.json").open("w", encoding="utf-8") as stream:
        json.dump(selection, stream, indent=2)
        stream.write("\n")

    ready_config = copy.deepcopy(config)
    for algorithm, choice in selected.items():
        ready_config["algorithms"][algorithm].update(choice["parameters"])
    ready_config["tuning_provenance"] = selection
    with (output / "summ_tuned.json").open("w", encoding="utf-8") as stream:
        json.dump(ready_config, stream, indent=2)
        stream.write("\n")
    test_config = copy.deepcopy(ready_config)
    test_config["dataset_names"] = [
        name for name in ready_config.get("dataset_names", [])
        if name not in datasets
    ]
    with (output / "summ_tuned_test.json").open("w", encoding="utf-8") as stream:
        json.dump(test_config, stream, indent=2)
        stream.write("\n")
    return selection


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune GAHGC/DHCC without labels by minimizing normalized HyDRA "
            "summary storage cost on development datasets."
        )
    )
    parser.add_argument("--defaults", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--algorithms", nargs="+", choices=["gahgc", "dhcc"],
        default=["gahgc", "dhcc"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--search",
        choices=["compact", "paper"],
        default="compact",
        help=(
            "compact runs 7 GAHGC and 16 DHCC candidates; paper runs 27 "
            "and 490 candidates, respectively"
        ),
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="discard the existing tuning_runs.csv instead of resuming it"
    )
    parser.add_argument(
        "--keep-summaries", action="store_true",
        help="retain every candidate summary (large); default keeps only scores"
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    with open(args.defaults, encoding="utf-8") as stream:
        defaults = json.load(stream)
    with open(args.config, encoding="utf-8") as stream:
        overrides = json.load(stream)
    config = _deep_merge(defaults, overrides)
    if args.search == "paper":
        print(
            "Warning: the paper grid contains 27 GAHGC and 490 DHCC "
            "candidates before multiplying by datasets and seeds.",
            flush=True,
        )
    selection = run_tuning(
        config=config,
        datasets=args.datasets,
        algorithms=args.algorithms,
        seeds=args.seeds,
        output_dir=args.output_dir,
        search=args.search,
        restart=args.restart,
        keep_summaries=args.keep_summaries,
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
