# Dataset and scope

**Source:** [Global House Purchase Decision Dataset — Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/global-house-purchase-decision-dataset), by `mohankrishnathalla`.

The originally provided notebooks loaded `global_house_purchase_dataset.csv`, reporting 200,000 observations with 25 columns. We exclude `property_id` (identifier) and use `decision` as binary target. **This is purchase-decision classification, not home-price regression.** The `price` column is an input predictor rather than the target.

- Target `decision`: 0 = no purchase, 1 = purchase.
- 19 numerical inputs, 4 categorical inputs (`country`, `city`, `property_type`, `furnishing_status`).
- Original notebook's label distribution: 153,932 negative (76.966%), 46,068 positive (23.034%).
- Split: stratified training 70%, validation 10%, final test 20%. All fits after splitting.
- We do **not** include dataset rows or preprocessed CSVs. Download from the data publisher and review its current terms/license before redistribution or commercial use.

### Data provenance / responsible use

Some fields describe buyer financial circumstances (salary, loans, expenses) and are used only in this educational exercise. No patient data are needed for this dataset. Do not enter personally identifying data in the demo. Sampling patterns or synthetic data generation logic can create shortcuts: near-perfect accuracy on this dataset does not establish performance on real purchases.

### Leakage audit

The original notebooks called `OneHotEncoder.fit_transform` and `StandardScaler.fit_transform` on the **entire dataset** before creating train/test/val splits. Optuna trials then trained with `validation_data=(X_test, y_test)` and maximized the same test accuracy. This makes the original `test` results unsuitable as final independent holdout evidence.

This cleaned repository fixes both implementation problems. The target also may be mechanically related to predictors, so a follow-up study should inspect source generation, leakage features and out-of-country generalization. We make no claim that this dataset represents deployed real-world purchase decisions.
