# Independent review of paired input-extent cohorts

Reviewed `make_sensitivity_cohort.py`, its focused tests, the reconstruction helpers,
and all four `evaluation/sensitivity-v1` outputs. No generator, input or runtime files
were changed during this review.

All four generators' actual output pairs preserve frozen partition IDs and order,
efficacy labels, and every column except the intended sequence fields. Independent
comparison against the frozen partitions and eligibility tables passes for every
row. Each eligible record occurs in exactly one train/validation/test/HeLa partition.
All declared input and output SHA256 values match. Local59/61 contexts preserve the
entire standardized57nt center; recovered21nt guides preserve the19nt core.

| Paired sensitivity | Train | Validation | Test | Eligible HeLa | Total |
|---|---:|---:|---:|---:|---:|
| AttSiOff21/59 | 1556 | 451 | 517 | 72 | 2596 |
| ENsiRNA61 context | 1815 | 449 | 618 | 1038 | 3920 |
| GNN4siRNA full mRNA | 1814 | 379 | 626 | 942 | 3761 |
| siRNADiscovery full mRNA | 1814 | 379 | 626 | 911 | 3730 |

The siRNADiscovery9756nt ceiling removes31 otherwise eligible HeLa records and no
training, validation or test records. Pairing applies the exclusion to both input
variants, so comparison IDs remain identical. Full-mRNA reconstruction retains
candidate-reference uncertainty in its audit and should not be described as proof
of the exact historical assay isoform.

The AttSiOff HeLa subset is only72/1047 (6.9%):36Harborth,34Ui-Tei and2Shabalina
records. Its sensitivity results describe this eligible subset, not full HeLa
transfer. Eligibility also differs across training/validation/test; report the
counts alongside results rather than generalizing effect size to the full dataset.

A provenance limitation remains in the original-overhang description. File007
lookup selects the uniquely listed21nt strand matching a19nt core without matching
publication/experimental metadata. In the retained cohort,54benchmark rows labeled
hsieh match File007's Authors=Reynolds; other cross-author and duplicate-conflict
cases also occur. This is an attribution discrepancy requiring interpretation,
not evidence that sequence pairing is wrong. Call these experimentally listed,
reconstructed21nt guides unless same-assay provenance is established independently.
Do not silently change the frozen labels based on this discrepancy.

Evidence: `audit/oligo_bert/paired_cohort_review_counts.csv`,
`paired_cohort_review_source_cell_counts.csv`, and
`attsioff_overhang_source_crosscheck.csv`. The generator explicitly warns that copied
57nt feature columns are provenance only: model preparation must recompute any
sequence-dependent features from the selected input variant. The inspected AttSiOff
and ENsiRNA preparation paths construct their model inputs from variant sequences.
