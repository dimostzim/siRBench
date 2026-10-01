# Historical conflict and context-variant evidence

Read-only follow-up on 2026-09-21; no data, labels, grouping or splits changed.

The first available repository commit, `706744f` (history-reset initialization),
already contains 4,098 unique guides in `data/siRBench_full.csv`. Every guide/context
pair and every rounded efficacy agrees with the later released 4,098-row union.
The current retained conflict choices therefore predate the available git history.
The initial README and tracked Python files contain no recovered row-conflict or
context-exclusion selection policy. Feature names called “duplicates” in
`make_all_features.py` concern redundant feature columns, not data records.

Eight Shabalina guide/context pairs absent from the release each have their exact
19-nt guide retained in another source with different 57-nt flanking context:
one Vickers, six Harborth and one Amarzguioui record. These observations establish
discarded context variants; they do not establish that the underlying guide was
excluded. Exact paired evidence is saved centrally in
`audit/provenance/context_variant_evidence.csv`.

Each of the eight discrepant Mix source rows belongs to a group containing two
Mix rows and one Shabalina row with the same guide/context pair. The retained
label matches one Mix row after rounding. Four retained Mix rows occur earlier
and four later than their conflicting partner in the archived file, so a universal
“keep first” or “keep last” rule does not explain these selections in that order.
The values likewise do not establish a universal maximum/minimum/mean policy.
The retained source labels are Khvorova (four), Ui-Tei  (three) and Hsieh  (one).
All source candidates are preserved in `audit/provenance/mix_conflict_evidence.csv`.

No documented historical conflict rule was recovered. The response/manuscript
should distinguish the observable retained choices from an author-confirmed
curation policy; the latter still requires the historical notebook or author
recollection. This finding alone does not justify changing retained labels.

## Author recollection and consistency check

The author tentatively recalled preferring a cell line or experiment with fewer
members. Archived Mix rows lack original constituent-source columns, but their
contiguous source blocks can be reconstructed from the released source labels of
unique, non-conflicting guides. CSV lines 2–47 map to Amarzguioui (46 rows),
48–91 Harborth (44), 92–105 Khvorova (14), 106–285 Reynolds (180), 286–345 Hsieh (60),
346–397 Ui-Tei (52), 398–473 Vickers (76). These are counts in this particular Mix
file before conflict removal; they are not the original publications' total sample
counts, which can differ.

Under that explicitly inferred block assignment, all eight retained choices favor
the smaller constituent source: Khvorova 14 over Reynolds 180 (one) or Hsieh 60 (three);
Ui-Tei 52 over Reynolds 180 (two) or Hsieh 60 (one); Hsieh 60 over Reynolds 180 (one).
The evidence table is `audit/provenance/rarer_source_policy_evidence.csv`.
This makes the author's experiment-frequency recollection conditionally consistent
with all eight choices. It does not independently establish the original losing
rows' experiment identities or prove the historical selection code. Pooled cell-line
frequency alone is not a universal explanation: three retained Ui-Tei/HeLa choices
belong to the larger pooled HeLa group. Labels remain unchanged.

A separate earlier Shabalina boundary removes three guides between the 653-row
`shabalina_dataset_final_withLaminA.csv` and the 650-row Final table. All three have
an eight-character `XXXXXXXX` placeholder context in the intermediate. A different
archived table (`shabalina_complete.csv`) contains candidate 57-nt contexts, so the
reason for their replacement was not recovered. Details are in
`audit/provenance/shabalina_prefinal_missing_contexts.csv`. These records were not
added to the revised corpus.
