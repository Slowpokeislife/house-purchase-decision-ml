"""Dataset loading and train-only fit preprocessing (no leakage)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, DATA_FILENAME, DATASET_SLUG, DEFAULT_DATA, FEATURES, NUMERICAL, SEED, TARGET


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def download_dataset(destination: Path = DEFAULT_DATA) -> Path:
    """Download user-authorized original data from Kaggle, not into git."""
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError('Install training extras: pip install -e ".[train]"') from e
    destination = Path(destination)
    if destination.exists():
        print(f"Dataset already exists: {destination}")
        return destination
    downloaded_folder = Path(kagglehub.dataset_download(DATASET_SLUG))
    source = downloaded_folder / DATA_FILENAME
    if not source.is_file():
        matches = list(downloaded_folder.rglob(DATA_FILENAME))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected {DATA_FILENAME} in {downloaded_folder}")
        source = matches[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def load_data(path: Path = DEFAULT_DATA) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run: house-purchase download (or supply --data)")
    df = pd.read_csv(path, low_memory=False)
    required = set(FEATURES + [TARGET])
    absent = sorted(required - set(df.columns))
    if absent:
        raise ValueError(f"Missing columns: {absent}. Check the original Kaggle schema.")
    if df[TARGET].isna().any():
        raise ValueError("Missing target labels are not supported")
    targets = set(df[TARGET].unique())
    if not targets.issubset({0, 1}) or len(targets) != 2:
        raise ValueError(f"Expected a binary target coded 0/1, got {targets}")
    return df


def split_data(df: pd.DataFrame, seed: int = SEED) -> DatasetSplits:
    """Stratified 70/10/20 split; no transformer ever sees val/test during fitting."""
    X, y = df[FEATURES].copy(), df[TARGET].astype(int).copy()
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold, y_hold, test_size=2/3, random_state=seed, stratify=y_hold
    )
    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def make_preprocessor() -> ColumnTransformer:
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num, NUMERICAL),
        ("cat", cat, CATEGORICAL),
    ], remainder="drop", sparse_threshold=0)


def prepare_data(path: Path = DEFAULT_DATA, seed: int = SEED):
    s = split_data(load_data(path), seed)
    preprocessor = make_preprocessor()
    train = np.asarray(preprocessor.fit_transform(s.X_train), dtype=np.float32)
    val = np.asarray(preprocessor.transform(s.X_val), dtype=np.float32)
    test = np.asarray(preprocessor.transform(s.X_test), dtype=np.float32)
    return s, preprocessor, train, val, test


def training_input_schema(train: pd.DataFrame) -> dict:
    """Aggregated ranges/modes for a data-free demo; no individual records."""
    schema = {"numerical": {}, "categorical": {}, "feature_order": FEATURES}
    for col in NUMERICAL:
        clean = pd.to_numeric(train[col], errors="coerce").dropna()
        schema["numerical"][col] = {
            "min": float(clean.min()), "max": float(clean.max()),
            "median": float(clean.median()),
        }
    for col in CATEGORICAL:
        clean = train[col].dropna().astype(str)
        schema["categorical"][col] = {
            "options": sorted(clean.unique().tolist()),
            "default": str(clean.mode().iloc[0]),
        }
    return schema
