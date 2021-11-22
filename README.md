# UnsupervisedFrameworkForComparingGraphEmbeddings

## Software prerequistes
To run the experiments you need:
* Python (install packages with `pip install -r requirements.txt`)
* Julia 1.6+ (install [CGE.jl](https://github.com/KrainskiL/CGE.jl) and [ABCDGenerator.jl](https://github.com/bkamins/ABCDGraphGenerator.jl/) as outlined in the repositories)
* You may need to rebuilt `lfrbench_udwov` yourself if provided binary is incompatible with your setup (built on macOS). For more information visit [LFR repository](https://github.com/eXascaleInfolab/LFR-Benchmark_UndirWeightOvp)


## Instructions
Run scripts in each folder in order defined by the numbering.

**Main experiments**

* `1_generate_graphs_and_embeddings.py` creates SBM and two LFR graphs used in the majority of experiments. After graphs creation, script produces HOPE and Node2Vec embeddings with various parameters
* `2_exact_approximate_scores.py` runs the embedding framework with exact and approximate algorithms and stores the results
* `3_correlation_ratio_results.py` calculates embedding scores for multiple number of landmarks as an input for correlation and ratio plots
* `4_comm_detect_clustering.py` runs community detection and clustering on produced embeddings
* `5_link_prediction.py` produces graphs with removed edges, generates embeddings on new graphs and run link prediction algorithm in multiple settings

Results produced by the scripts are used in `Visualizations.ipynb` notebook. Execution of the scripts require substantial computing power and we recommend to adjust the code for distributed computing (e.g. by running separate subsets of the loops on separate nodes)

**ABCD experiments**

To create the ABCD graph run `1_generate_abcd_graph.sh` (requires `ABCDGenerator.jl`). Run `2_abcd_experiments.py` to produce embeddings for the ABCD graph and results for rescaling and rewiring experiments (multiple modifications of the graph and embeddings will be produced). Once you receive the results, plots may be produced with `Visualization_ABCD.ipynb` notebook.

**College Football layouts**
Run `Football_layouts.ipynb` to produce embeddings and plots for College Football graph.
