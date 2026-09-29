# Overview
This repository presents the code for **HyDRA**, a novel framework for finding a lossless summary of a possibly weighted hypergraph by seeking to minimize a storage-aware objective function that quantifies the space required to store the summary hypergraph's super-hyperedges and the *correction table* that enables exact reconstruction of the original hypergraph.

**HyDRA** is built upon an existing information-theoretic agglomerative co-clustering algorithm called CoClusLSH [1], which greedily merges similar row
and column clusters, identified via locality-sensitive hashing (LSH).

# Content
    data/            ... datasets used in the experimental evaluation
    code/            ... Python source files and scripts
    config/          ... configuration files for running the experiments
    environment.yaml  ... required Python libraries

## Source Files
The *code* folder includes the following files:

 - *hyp_sum.py*: implementation of HyDRA.
 - *leman.py*: implementation of CoClusLSH.
 - *gahgc.py*: independent, assumption-explicit implementation of GAHGC and
   conversion of its labels to the common summary-construction contract.
 - *dhcc.py*: independent implementation of Manifold-aware Dual Hypergraph
   Co-Clustering (DHCC), with sparse row- and column-side hypergraphs.
 - *helpers.py*: helper functions used by CoClusLSH and HyDRA.
 - *summary_utils.py*: some useful methods to process, read, and write summaries of hypergraphs. 
 - *utils.py*: some useful methods.
 - *plots_utils.py*: functions to plot an incidence matrix whose rows and columns are reordered based on the cluster assignments.
 - *hn\_moea\_im*: Python library to run Influence Maximization on hypergraphs and hypergraph summaries.
 - *pagerank*: Python library to run PageRank on hypergraphs.
 - *taucc*: Python library to run the co-clustering algorithm PB-$\tau$-CC.
 - *tautcc*: Python library to run the co-clustering algorithm PB-$\tau$-TCC.
 - *queries.py*: methods to run several types of queries on hypergraphs and hypergraph summaries.
 - *run_data_summarization.py*: script to perform lossless hypergraph summarization.
 - *run_hyper_im.py*: script to perform Influence Maximization on hypergraphs and hypergraph summaries.
 - *run_queries.py*: script to answer queries on hypergraphs and hypergraph summaries.
 
## Config Files
The *config* folder includes the following *json* files to set the values of the hyperparameters used by HyDRA and the competitors:

 - *defaults.json*: includes the full list of datasets and algorithms, the paths to the data and output directories, and default values of the hyperparameters of the summarization algorithms. For each dataset, it reports the default number of node and hyperedge clusters to search, and whether it is weighted or not. For each algorithm that requires a number of clusters as input, it reports the default number of clusters. For CoClusLSH and HyDRA, it reports the default cost threshold.
 - *im.json*: includes the list of datasets and algorithms to consider in the Influence Maximization experiments, as well as the hyper-parameters used by the *hn\_moea\_im* library. To overwrite default values, the desired parameter configuration must be included in this file.
 - *queries.json*: includes the list of datasets and algorithms to consider in the query-answering experiment, as well as the parameters required to run each type of query. To overwrite default values, the desired parameter configuration must be included in this file.
 - *summ.json*: includes the list of datasets and algorithms to consider in the hypergraph summarization experiment. To overwrite default values, the desired parameter configuration must be included in this file.

## Parameters

#### Parameters used by CoClusLSH and HyDRA in hypergraph summarization

- *r*: signature size for LSH (used by CoClusLSH and HyDRA).
- *b*: number of hash tables for LSH (used by CoClusLSH and HyDRA).
- *weights*: importance of each term in the storage-aware objective function used by HyDRA.
- *min_size*: min size of a candidate group of clusters to merge in order to be processed (used by CoClusLSH and HyDRA).
- *max_trials*: max number of cluster merges to try (used by CoClusLSH and HyDRA).
- *max\_no\_improvements*: max number of merge attempts with no improvements before exiting (used by CoClusLSH and HyDRA).

#### Parameters used by GAHGC

- *default_nclusters*: number of sample clusters when `use_means` is false.
- *alpha*: weight of within-hyperedge label-disagreement penalties.
- *max\_walk\_length*, *n_walks*: weighted hypergraph random-walk settings.
- *embedding_dim*, *window*, *negative_samples*, *embedding_epochs*,
  *learning_rate*, *batch_size*: skip-gram negative-sampling settings.
- *max_iter*: maximum number of clustering updates.
- *bandwidth*: Equation 9 kernel bandwidth; it can also be set to `"median"`.
- *embedding_backend*: `"sgns"` for the paper's skip-gram stage or
  `"ppmi_svd"` for a faster DeepWalk-style sensitivity run.

The GAHGC article [4] does not specify several embedding settings, and Equations
18--20 do not define a consistent executable center update. The included
`gahgc.py` is therefore an independent reimplementation: it preserves the
input hypergraph topology, follows the weighted-walk and dominant-feature-label stages, and uses an objective-faithful solver for Equation 7.

The checked-in starting values are `alpha=0.5`, `max_walk_length=50`, and
`n_walks=50`: the first is the paper's selected value and the latter two are
the lower bounds of its reported stable ranges. Because the paper does not
report the Equation 9 bandwidth, `bandwidth="median"` selects a scale from the
data without labels. The unspecified SGNS settings use standard fixed values
(dimension 32, window 5, five negatives, one epoch, and learning rate 0.025).

#### Parameters used by DHCC

- *default_nclusters*, *default\_n\_column\_clusters*: row and column cluster
  counts when `use_means` is false. With `use_means` true, both values come
  from each dataset's `means_nclusters` pair.
- *sample_regularisation*, *feature_regularisation*: the paper's lambda and mu
  manifold penalties.
- *n_neighbors*: k in the row- and column-side k-nearest-neighbour
  hypergraphs.
- *max_iter*, *tol*: maximum alternating updates and relative stopping
  tolerance.
- *metric*: neighbour-search metric; `euclidean` reproduces the paper's
  distance definition.
- *kmeans\_n\_init*, *kmeans\_max\_iter*: Algorithm 1 initialization settings.
- *ridge*, *floor*: numerical safeguards for the core inverse and
  multiplicative updates.
- *n_jobs*: parallelism used by neighbour search.

The paper [3] searches `n_neighbors` from 1 through 10 and each regularisation
coefficient over powers of ten from `1e-3` through `1e3`; its final
cross-dataset choices are `n_neighbors=3` and both coefficients equal to 100.
These are the checked-in starting defaults. The implementation follows
Equations 2, 10, 13, 21, 24, and 32 and is labeled `DHCC-reimpl` because no
official author code was located.

#### Parameters used in query-answering

- *alpha*: PageRank damping factor.
- *version*: where PageRank is to be computed (possible values are *Hypergraph*, *Graph*, *Bipartite*)
- *num_iter*: number of iterations for PageRank.
- *sample_size*: sample size used to generate node and node pair sets for reachability and PageRank experiments.
- *query_exp*: binary vector to specify which queries to run: [PageRank, Degree, Reachability, Connected Components, h-hop Neighborhood, Closeness Centrality].

#### Parameters used by HN-MOEA-IM

- *min_seed_nodes*: min seed set size for EA algorithm.
- *max_seed_nodes*: max seed set size for EA algorithm.
- *population_size*: population size for EA algorithm.
- *offspring_size*: for EA algorithm.
- *max_generations*: for EA algorithm.
- *tournament_size*: for EA algorithm.
- *mutation_rate*: mutation rate.
- *crossover_rate*: crossover rate.
- *num_elites*: for EA algorithm.
- *threshold*: used for LT contagion model (varies per network).
- *p_min*: used for SICP contagion model.
- *p_max*: used for SICP contagion model.
- *max_hop*: max number of hops within which influence is propagated.
- *model*: contagion model (can take values *WC*, *LT*, or *SICP*).
- *init_strategy*: strategy to initialize the population.
- *no_simulations*: number of Monte-Carlo simulations.
- *custom_mutation*: for EA algorithm.

#### Other parameters

- *algorithm_names*: list of algorithms to consider in the experiment. It can include:
    - *CoClusLSH*: algorithm from [1].
    - *HyDRA*: our proposed algorithm.
    - *Random-True*: random clustering with clusters of equal size.
    - *Random-False*: random clustering with clusters of random size.
    - *mcc*: CoclustMod algorithm (from *coclust* Python library).
    - *smcc*: CoclustSpecMod algorithm (from *coclust* Python library).
    - *itcc*: CoclustInfo algorithm (from *coclust* Python library).
    - *cc*: PB-$\tau$-CC algorithm from [2].
    - *tcc*: PB-$\tau$-TCC algorithm from [2].
    - *gahgc*: independent implementation of the General Adaptive Hypergraph
      Model for Coclustering [4].
    - *dhcc*: independent implementation of Manifold-aware Dual Hypergraph
      Co-Clustering [3].
    - *sc*: SpectralCoclustering algorithm (from *sklearn* Python library).
    - *sb*: SpectralBiclustering algorithm (from *sklearn* Python library).
- *nruns*: number of independent runs to perform in hypergraph summarization or query-answering.
- *max_workers*: number of parallel executions.
- *verbose*: print details.
- *seed*: seed for reproducibility
- *sep*: vertex separator used in the input files containing the original hypergraphs.

## Label-free tuning of the reimplementations

`tune_reimplementations.py` selects one global configuration per algorithm on
explicitly designated development datasets. It uses no class labels and
minimizes the dataset-balanced mean of the framework's normalized storage
objective, `Cost / Orig. Cost`. The default compact search evaluates 7 GAHGC
and 16 DHCC candidates with three seeds and checkpoints after every run:

```sh
python tune_reimplementations.py \
  --defaults ../config/defaults.json \
  --config ../config/summ.json \
  --datasets senate dblp \
  --seeds 0 1 2 \
  --output-dir ../out/reimplementation_tuning
```

Rerunning the same command resumes completed runs. The output contains
per-run and per-candidate CSV audit tables, `selected_settings.json`, and a
complete `summ_tuned.json`. It also writes `summ_tuned_test.json`, whose
`dataset_names` excludes the development datasets and which can be passed
directly to the summarization runner for the held-out comparison. Use
`--search paper` only when the full 27-candidate GAHGC and 490-candidate DHCC
searches are computationally feasible. The development datasets should be
selected before inspecting results and should preferably be excluded from the
final aggregate comparison.

## Practical scalability experiment

`run_scalability.py` measures HyDRA while increasing the number of complete
hyperedges retained from each input. Samples are nested within each
dataset/seed, and isolated nodes are removed. The runner checkpoints after
every execution and records input dimensions, incidence nonzeros, wall-clock
time, merge rounds, final cluster counts, and cost improvement:

```sh
python run_scalability.py \
  --defaults ../config/defaults.json \
  --config ../config/summ.json \
  --datasets classic house \
  --fractions 0.2 0.4 0.6 0.8 1.0 \
  --seeds 0 1 2 \
  --output ../out/hydra_scalability.csv
```

# Input Format
The file containing the hyperedges of the original hypergraph should contain one hyperedge per row.

- Unweighted Hypergraphs: node\_1 {separator} node\_2 {separator} ... node\_k
- Weighted Hypergraphs: node\_1 {separator} node\_2 {separator} ... node\_k {separator} weight

The separator symbol and information about whether a hypergraph is weighted or not must be specified in the configuration file *defaults.json*.

Node IDs will be remapped in the range [0, ..., n - 1].

The file extension is assumed to be *.csv*.

The folder data includes some of the datasets used in our experimental evaluation.

# Queries

The configuration file *queries.json* allows the user to run the following queries on the summaries generated by the summarisation algorithms:

1. PageRank
2. Degree and Hyperdegree
3. Reachability
4. Connected Components
5. k-hop Neighbors
6. Closeness Centrality

To run a query job:

```sh
python run_queries.py \
  --defaults ../config/defaults.json \
  --config ../config/queries.json
```

# Requirements
To run the Python scripts in the folder *code*, you must install the libraries listed in *environment.yaml*.

A conda environment can be easily created by running the following commands:

```sh
conda env create -f environment.yaml
conda activate hypsum
```

# License
This package is released under the GNU General Public License.

# References

[1] Tiantian Gao and Leman Akoglu. 2014. Fast information-theoretic agglomerative co-clustering. In Australasian Database Conference. Springer, 147–159.

[2]  Elena Battaglia, Federico Peiretti, and Ruggero G Pensa. 2024. Fast parameterless prototype-based co-clustering. Machine Learning 113, 4 (2024), 2153–2181.

[3] Yi Song, Hongjun Wang, Luqing Wang, Xu Li, and Tianrui Li. 2026.
Manifold-aware dual hypergraph co-clustering. Neurocomputing 671, 132680.
https://doi.org/10.1016/j.neucom.2026.132680

[4] X. Wang, H. Wang, and T. Li. General adaptive hypergraph model for coclustering. IEEE Transactions on Computational Social Systems, 13(1):166–179, 2025.
