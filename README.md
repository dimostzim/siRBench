# siRBench

Data and code for benchmarking siRNA efficacy predictors.

- [data/](data/README.md): Download and prepare the dataset, features and splits.
- [benchmark/](benchmark/README.md): Run the published predictors and baselines, and evaluate predictions.
- [models/](models/README.md): Train and run the five Agentomics models and TabPFN.

The published predictors use Docker. Data scripts, Agentomics and TabPFN use uv.

Datasets, training outputs, predictions and the experimental code snapshot are in
[Zenodo version 2](https://doi.org/10.5281/zenodo.23001225).

First-party code is MIT licensed. External implementations, model weights and
source datasets retain their original terms. TabPFN weights, OligoFormer source
and Rosetta are obtained separately under their respective licences.
