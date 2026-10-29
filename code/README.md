# Scientific Implementations

These files preserve the original scientific implementations and their local
dependencies. Keeping historical helper names avoids changing frozen source
hashes or silently altering the experiments.

| Main implementation | Analysis |
| --- | --- |
| `run_repeated_holdout.py`, `_shared/repeated_holdout.py` | PCA7 repeated-holdout experiment |
| `run_repeated_holdout_pca9.py`, `_shared/repeated_holdout_pca9.py` | PCA9 dimensional sensitivity |
| `run_professor_followup.py`, `_shared/professor_followup.py` | Random-coordinate and matched-backbone controls; risk calibration diagnostics |
| `run_risk_only_ablation.py` | Inherited-setting Risk-only component comparison |
| `run_budget_riskonly_followup.py` | Restricted search and independently tuned Risk-only |
| `analyze_matched_geometry_20261028.py` | Fixed-setting actual/shuffled Gram geometry |
| `analyze_fixed_model_cost_ratios_20261029.py` | Fixed-model threshold sensitivity |

The `_shared/` modules and older runner/protocol names include dependencies used
by these implementations. They are not additional independent datasets or new
confirmatory experiments.

The original runners read archived preparations, selected models and immutable
run receipts; some also enforce the original container path. The portable,
data-free entry point for this repository is `analysis/reproduce_aggregates.py`.
It recomputes the supplied aggregate statistics without invoking these runners.
