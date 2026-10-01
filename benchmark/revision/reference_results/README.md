# Fold-specific siRBench reference-model results

This analysis adds the five accepted, validation-selected Agentomics pipelines to the completed benchmark. It does not rerun training or model search. `sirbench_reference` denotes the foldwise search procedure; the five saved models can have different architectures.

There is one selected fit per grouped fold, with no repeated search or three-seed refitting and no random-split Agentomics experiments. Grouped metrics pool exactly 3,051 unique out-of-fold predictions. HeLa metrics average the five individual model metrics over the same 1,047 records; predictions are not averaged. The aligned HeLa sensitivity is a subset of those existing predictions.

The analysis imports the existing metric/bootstrap implementation. It uses 2,000 identical group draws with seed 20260921, clustered by the 43 target/guide components or 45 HeLa target groups. Confidence intervals are conditional on saved models and fixed partitions. Paired differences against combined ridge, ENsiRNA, siRNADiscovery and the two TabPFN variants are exploratory and unadjusted. Unequal training replication and completed search iterations are explicit limitations. No superiority claims are inferred from point ranks.

Verification covers prediction IDs, complete per-fold cohorts, label agreement with the corrected dataset, finite outputs, and exact reproduction (absolute tolerance 1e-12) of the existing comparator estimates and bootstrap intervals. Existing results are copied unchanged before appending the reference-model row. `training_seed=-1` is the unreplicated-fit sentinel, not the internal random seed of each saved pipeline.

## Reproduce

From the repository root, follow [the portable result replay](../../../REPRODUCING.md#numerical-results-and-baselines).
The archive's `reference-evaluation/` contains all five models' input/prediction cohorts.
The frozen analysis implementation in `code/` is preserved without numerical changes.
Generated tables are kept in Zenodo rather than duplicated in the source checkout.
