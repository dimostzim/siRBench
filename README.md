# siRBench

Repository layout:

- `benchmark/`: Docker-based predictor prepare/train/test wrappers, baselines and evaluation scripts. See [benchmark setup](benchmark/README.md).
- `data/`: Dataset reconstruction, feature generation and grouped/random splits. See [data scripts](data/scripts/README.md).
- `models/`: The five Agentomics models and frozen/fine-tuned TabPFN, with uv environments and training/inference scripts. See [model setup](models/README.md).

Datasets, training outputs, predictions and the experimental code snapshot are in
[Zenodo version 2](https://doi.org/10.5281/zenodo.23001225).

First-party code is MIT licensed. External implementations, model weights and
source datasets retain their original terms. TabPFN weights, OligoFormer source
and Rosetta are obtained separately under their respective licences.
