# Risk-Conditioned Kernel Learning

Research code and aggregated results for cost-sensitive credit classification
with classical RBF and simulated quantum fidelity kernels.

The study examines three questions: whether customer-specific risk scores are
useful, whether the base kernel adds value beyond those scores, and how changes
in kernel geometry relate to classification cost. The main constructions are
XGBoost Soft Gate and Rank Append.

## Repository Layout

| Directory | Contents |
| --- | --- |
| `code/` | Original experiment, kernel, reporting and validation implementations, with required historical dependencies |
| `analysis/` | Portable recomputation of the supplied aggregate statistics |
| `results/` | Seed-level classification summaries, paired differences, complete comparison families, geometry and runtime summaries |
| `docs/` | Reproduction instructions, analysis definitions and frozen protocols |
| `environments/` | Original server dependency lock and environment descriptions |
| `data/` | Benchmark acquisition instructions; download destination |
| `tests/` | Tests for the portable aggregate replay |

`PUBLIC_MANIFEST.json` identifies the files in this code-and-aggregate release.
The source identifiers recorded in results describe the original experiment
lineage; `results/source_map.json` maps the supplied copies to that lineage.

## Reproduce the Aggregate Statistics

The portable replay was verified with Python 3.14.4, NumPy 2.4.3, pandas 3.0.1
and SciPy 1.17.1. It does not require quantum simulation or original datasets.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/reproduce_aggregates.py --output-dir /tmp/quantum-kernel-replay
python analysis/reproduce_runtime_components.py --output-dir /tmp/quantum-kernel-runtime
python -m unittest discover -s tests -v
```

The replay reconstructs all 632 paired comparisons from 6320 seed differences,
reapplies their seven complete original Holm families, checks the geometry
summaries, and checks the 1440 fixed-model cost-ratio aggregate rows. It does not
reselect thresholds from customer scores or train models.

## Scientific Protocol

- German Numeric, South German, HMEQ and Taiwan use ten shared seeds 10 through
  100 in increments of 10, with stratified 80/10/10 training/validation/test splits.
- German Numeric and South German are related representations of one data
  source, not independent populations.
- Training risk scores use five-fold cross-fitting. Models and validation-cost
  thresholds are fixed before each run's test evaluation.
- The principal cost is `5FN + FP`. Fixed-model 2:1 and 10:1 sensitivity changes
  validation thresholds while retaining models selected under 5:1.
- Random and Shuffled controls address different information comparisons.
  Random costs are averaged over three realizations within each seed.
- Matched Shuffled Geometry evaluates Gram matrices without fitting shuffled
  classifiers; its geometry summaries do not supply classifier costs.

See [Reproduction](docs/REPRODUCIBILITY.md) and
[Statistical Definitions](docs/STATISTICS.md) for scope and interpretation.
Original full experiment runners retain their completed-run safeguards and
archival dependencies. They are supplied as implementation records, not as a
new automatic campaign to launch after cloning.

## Environments and Data

The original fitting and GPU Gram environment differs from the portable
statistics environment. See [Environment Notes](environments/README.md) and
[Data Inventory](data/README.md). No source code is rewritten to imply that
the different environments or fitting procedures were identical.

## License

Original project code is distributed under the [MIT License](LICENSE).
Benchmark datasets retain their original providers' terms.
