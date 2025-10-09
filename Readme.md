# Overview
This repository presents the code for **HyDRA**, a novel framework to find a lossless summary of a weighted hypergraph.
This algorithm attempts to minimize a storage-aware objective function that quantifies the space required to store the super-hyperedges of the summary hypergraph and the *correction table* that permits the exact reconstruction of the original hypergraph.

**HyDRA** is built upon an existing information-theoretic agglomerative co-clustering algorithm called CoClusLSH [1], which greedily merges similar row and column clusters, identified via locality-sensitive hashing (LSH).

# Content
    data/            ... datasets used in the experimental evaluation
    code/            ... Python source files and scripts
    config/          ... configuration files for running the experiments
    environment.yaml  ... required Python libraries

## Source Files
The *code* folder includes the following files:

 - *hyp_sum.py*: implementation of HyDRA.
 - *leman.py*: implementation of CoClusLSH.
 - *helpers.py*: helper function used by CoClusLSH and HyDRA.
 - *summary_utils.py*: some useful methods to process, read, and write summaries of hypergraphs. 
 - *utils.py*: some useful methods.
 - *plots_utils.py*: functions to plot an incidence matrix whose rows and columns are reordered based on the cluster assignments.
 - *hn_moea_im*: Python library to run Influence Maximization on hypergraphs and hypergraph summaries.
 - *pagerank*: Python library to run PageRank on hypergraphs.
 - *taucc*: Python library to run the co-clustering algorithm PB- $\tau$ CC.
 - *tautcc*: Python library to run the co-clustering algorithm PB- $\tau$ TCC.
 - *queries.py*: methods to run several types of queries on hypergraphs and hypergraph summaries.
 - *run_data_summarization.py*: script to perform lossless hypergraph summarization.
 - *run_hyper_im.py*: script to perform Influence Maximization on hypergraphs and hypergraph summaries.
 - *run_queries.py*: script to answer queries on hypergraphs and hypergraph summaries.
 
 ## Config Files
The *config* folder includes the following *yaml* files to set the values of the hyperparameters used by HyDRA and the competitors:

 - *defaults.json*: includes the full list of datasets and algorithms, the paths to the data and output directories, and default values of the hyperparameters of the summarization algorithms. For each dataset, it reports the default number of node and hyperedge clusters to search, and whether the dataset is weighted or not. For each algorithm that requires a number of clusters as input, it reports the default number of clusters that it uses. For CoClusLSH and HyDRA, it reports the default cost threshold.
 - *im.json*: includes the list of datasets and algorithms to consider in the Influence Maximization experiments, as well as the hyperparameters used by the *hn_moea_im* library. To overwrite default values, the desired parameter configuration must be included in this file.
 - *queries.json*: includes the list of datasets and algorithms to consider in the query-answering experiment, as well as the parameters required to run each type of query. To overwrite default values, the desired parameter configuration must be included in this file.
 - *summ.json*: includes the list of datasets and algorithms to consider in the hypergraph summarization experiment. To overwrite default values, the desired parameter configuration must be included in this file.

## Parameters

Parameters used by CoClusLSH and HyDRA in hypergraph summarization:

- *r*: signature size for LSH (used by CoClusLSH and HyDRA).
- *b*: number of hash tables for LSH (used by CoClusLSH and HyDRA).
- *weights*: importance of each term in the storage-aware objective function used by HyDRA.
- *min_size*: min size of a candidate group of clusters to merge to be processed (used by CoClusLSH and HyDRA).
- *max_trial*s: max number of cluster merges to try (used by CoClusLSH and HyDRA).
- *max_no_improvements*: max number of merge tentative with no improvements before exiting (used by CoClusLSH and HyDRA).

Parameters used in query-answering:

- *alpha*: PageRank dumping factor.
- *version*: where PageRank is to be computed (possible values are *Hypergraph*, *Graph*, *Bipartite*)
- *num_iter*: number of iterations for PageRank.
- *sample_size*: sample size used to generate node and node pair sets for reachability and PageRank experiments.
- *query_exp*: binary vector to specify which queries to run: [PageRank, Degree, Reachability, Connected Components, h-hop Neighborhood, Closeness Centrality].

Parameters used by HN-MOEA-IM:

- *min_seed_nodes*: min seed set size for the EA algorithm.
- *max_seed_nodes*: max seed set size for the EA algorithm.
- *population_size*: population size for the EA algorithm.
- *offspring_size*: offspring size for the EA algorithm.
- *max_generations*: maximum number of generations for the EA algorithm.
- *tournament_size*: tournament size for the EA algorithm.
- *mutation_rate*: mutation rate.
- *crossover_rate*: crossover rate.
- *num_elites*: for the EA algorithm.
- *threshold*: used for the LT contagion model (varies per network).
- *p_min*: used for the SICP contagion model.
- *p_max*: used for the SICP contagion model.
- *max_hop*: max number of hops within which influence is propagated.
- *model*: contagion model (can take values *WC*, *LT*, or *SICP*).
- *init_strategy*: strategy to initialize population.
- *no_simulations*: number of Monte-Carlo simulations.
- *custom_mutation*: whether to use the custom mutation function for the EA algorithm.

Other parameters:

- *algorithm_names*: list of algorithms to consider in the experiment. It can include:
    - *CoClusLSH*: algorithm from [1].
    - *HyDRA*: our proposed algorithm.
    - *Random-True*: random clustering with clusters of equal size.
    - *Random-False*: random clustering with clusters of random size.
    - *mcc*: CoclustMod algorithm (from *coclust* Python library).
    - *smcc*: CoclustSpecMod algorithm (from *coclust* Python library).
    - *itcc*: CoclustInfo algorithm (from *coclust* Python library).
    - *cc*: PB-$\tau$-CC algorithm from [2].
    - *tcc*: PB-$\tau$-CC algorithm from [2].
    - *sc*: SpectralCoclustering algorithm (from *sklearn* Python library).
    - *sb*: SpectralBiclustering algorithm (from *sklearn* Python library).
- *nruns*: number of independent runs to perform in hypergraph summarization or query-answering.
- *max_workers*: number of parallel executions.
- *verbose*: print details.
- *seed*: seed for reproducibility
- *sep*: vertex separator used in the input files containing the original hypergraphs.

# Input Format
The file containing the hyperedges of the original hypergraph should contain one hyperedge per row.
- Unweighted Hypergraphs: node_1 {separator} node_2 {separator} ... node_k
- Weighted Hypergraphs: node_1 {separator} node_2 {separator} ... node_k {separator} weight

The separator symbol and information about whether a hypergraph is weighted or not must be specified in the configuration file *defaults.json*.

Node IDs will be remapped in the range [0, ..., n - 1].

The file extension is assumed to be *.csv*.

The folder data includes some of the datasets used in our experimental evaluation.

# Requirements
To run the Python scripts in the folder *code*, you must install the libraries listed in the file environment.yml.

A conda environment can be easily created by running the following commands:

```sh
conda env create -f environment.yaml
conda activate hypsum
```

# License
This package is released under the GNU General Public License.

# References

[1] Tiantian Gao and Leman Akoglu. 2014. Fast information-theoretic agglomerative co-clustering. In Australasian Database Conference. Springer, 147–159.

[2]  Elena Battaglia, Federico Peiretti, and Ruggero G. Pensa. 2024. Fast parameterless prototype-based co-clustering. Machine Learning 113, 4 (2024), 2153–2181.
