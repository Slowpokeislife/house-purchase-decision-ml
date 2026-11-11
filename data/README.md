# Data directory (not versioned)

The source dataset and any train/validation/test splits must **not** be committed here.

Get the dataset from: https://www.kaggle.com/datasets/mohankrishnathalla/global-house-purchase-decision-dataset

Use `house-purchase download` or manually place `global_house_purchase_dataset.csv` in `data/raw/`.
All generated paths are excluded by `.gitignore`.

Do not upload actual customer/patient records or secrets. The provided original ZIP contained preprocessed splits; they are intentionally excluded.
