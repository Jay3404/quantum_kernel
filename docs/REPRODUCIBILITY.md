# Reproduction Guide

## Portable Aggregate Replay

Run the commands in the root README from a clean clone. The aggregate replay
checks 632 original paired summaries, all 6320 paired differences and the seven
complete Holm families. It also checks the 160 matched-geometry configurations
and reaggregates 1440 fixed-model cost-ratio rows.

The error-cost sensitivity records contain model-level counts, thresholds and
metrics. This replay checks `cost = ratio * FN + FP` and the supplied mean/SD
summaries. It does not redo validation threshold optimization from individual
customer scores. All models in this sensitivity were originally selected at 5:1.

Runtime replay reconstructs 32 numeric component-summary cells from 160 recorded
SVM entries. Definitions and exclusions are in `docs/RUNTIME_ACCOUNTING.md`.
It performs no new timing measurements.

## Original Scientific Code

The implementations under `code/` preserve the experiment lineage and original
source bytes. Some older modules are transitive dependencies of the final
pipeline and are retained for that reason. `PUBLIC_MANIFEST.json` records the
released source hashes. The frozen JSON protocols under `docs/` retain their
original identifiers, settings and historical paths.

The full campaign runners require their archived parent-run inputs, models and
selection receipts and may require the original container layout. They are not
called by the aggregate replay or the public test suite. No scientific jobs are
started by cloning, installing the replay dependencies, or running these tests.

## Results Layout

- `results/pca7/` and `results/pca9/`: model-level seed summaries and full
  original comparison families.
- `results/controls/`: threshold, calibration, Random and inherited Risk-only
  diagnostics, aggregated by model and seed.
- `results/search_budget/`: restricted-search and independently tuned Risk-only
  results, including the full Holm64 family.
- `results/geometry/`: descriptive geometry changes and correlations.
- `results/cost_sensitivity/`: fixed-model 2:1/5:1/10:1 counts and tidy summaries.
- `results/runtime/`: recorded preparation and selected-model time components.
- `results/statistics/`: complete comparison index, normalized paired summaries
  and unrounded seed differences.

`results/source_map.json` maps each copied CSV to its original analysis
identifier and content hash. No numerical result is changed during organization.
