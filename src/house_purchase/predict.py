"""Inference on one raw record, using train-fitted transformations."""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd

from .config import FEATURES


def predict_one(artifacts: Path, record: dict):
    artifacts = Path(artifacts)
    missing = sorted(set(FEATURES) - set(record))
    if missing:
        raise ValueError(f"Missing input fields: {missing}")
    prep = joblib.load(artifacts / "preprocessor.joblib")  # Only load your own trusted artifacts.
    X = prep.transform(pd.DataFrame([{key: record[key] for key in FEATURES}]))
    kind = json.loads((artifacts / "metadata.json").read_text(encoding="utf-8"))["model_kind"]
    if kind == "baseline":
        model = joblib.load(artifacts / "model.joblib")
        probability = float(model.predict_proba(X)[0, 1])
    elif kind == "keras":
        from tensorflow import keras
        model = keras.models.load_model(artifacts / "model.keras")
        probability = float(model.predict(X, verbose=0).ravel()[0])
    else:
        raise ValueError(f"Unsupported model type {kind}")
    return {"predicted_purchase": bool(probability >= .5),
            "purchase_probability": probability, "threshold": .5}
