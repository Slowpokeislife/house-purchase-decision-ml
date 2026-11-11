# Model card — House Purchase Decision Predictor

## Intended use
Educational binary classification of the Kaggle `decision` field for a portfolio case study. **Not** a mortgage, underwriting, real-estate, insurance, or credit eligibility system.

## Models
- Logistic regression baseline with balanced class weights.
- Original Keras MLP: 128 → 64 → 32 → sigmoid, ReLU, dropout 0.2, L2=1e-4, first-layer batch normalization, Adam.
- Previously selected Optuna MLP: 224 → 64 → 48 → 48 → sigmoid, LeakyReLU(0.1), dropout 0.15, L2≈5.47e-5, Adam lr≈0.001967.
- Optional fresh Optuna hyperparameter optimization: layer count, widths, dropout, L2, batch normalization, LR.

## Evaluation protocol
- Stratified 70/10/20 with seed 42.
- All imputing, scaling, encoding fit on training subset only.
- Training callbacks and Optuna objective consult **validation only**.
- Final test is scored once after model selection in each experiment.
- Report accuracy, balanced accuracy, precision, recall, F1, ROC AUC, average precision, confusion matrix. No comparison is valid if experiments repeatedly tune against the final test.

## Known limitations
The source dataset may have predictable/mathematically generated targets. Neither original nor new results establish calibration, temporal stability, fairness across groups, or performance in a real housing market. The inputs include financial and geographic variables; do not use to make decisions about individuals.

## Privacy and security
No dataset rows, account names, OAuth redirect URLs, API credentials, original notebooks with embedded outputs, or local training artifacts are published. Models must be retrained locally before the demo works. Joblib/pickle is unsafe to load from untrusted sources.
