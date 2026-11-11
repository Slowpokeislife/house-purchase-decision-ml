# Experimental results: archive versus reproducible protocol

## Historical outputs from the user's original notebooks

| Historical item | Evidence in original notebook | Caution |
| --- | --- | --- |
| Class distribution | 153,932 no purchase / 46,068 purchase | Strong class imbalance (~23% positive) |
| Baseline test confusion matrix | `[[30780, 7], [0, 9213]]` (40,000 rows) | Its `test` split served as training validation |
| Optimized test confusion matrix | `[[30787, 0], [0, 9213]]` (40,000 rows) | Same `test` split used for tuning/early stopping |
| Optimized 10% holdout matrix | `[[15393, 0], [0, 4607]]` (20,000 rows) | Not used directly for Optuna, but preprocessing was fit before splitting |
| Original search budget | 20 Optuna trials configured | Trial history/result files not supplied, so not independently audited |

Historical figures are displayed in `assets/legacy/` and **must not** be interpreted as a new leakage-free benchmark. The near-perfect metrics may reflect shortcut relationships in the dataset. Avoid headline claims of 100% accurate real-world predictions.

## New reproducible results

No new performance numbers are claimed: running the supplied clean pipeline on the Kaggle dataset is required. `artifacts/*/metrics.json` will contain test metrics, `validation_metrics.json` will contain validation metrics, and `curves.png` will contain training curves.

**Suggested experimental extension:** compare full feature set against removal of financial/decision-adjacent variables; assess PR-AUC (positive class ~23%), threshold calibration, and shift across countries. Do not tune on final test.
