"""Evaluation, plot export, local optional MLflow logging."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def metrics_at_threshold(y_true, probabilities, threshold=.5) -> dict:
    y_true = np.asarray(y_true).ravel()
    prob = np.asarray(probabilities).ravel()
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, pred, output_dict=True, zero_division=0),
    }


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def save_learning_curves(history: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("loss", "pr_auc")):
        ax.plot(history.get(metric, []), label="Train")
        ax.plot(history.get("val_" + metric, []), label="Validation")
        ax.set(xlabel="Epoch", ylabel=metric, title=metric.upper())
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def maybe_log_mlflow(run_name: str, artifact_dir: Path, metrics: dict, params: dict, enabled: bool) -> None:
    """Tracking is opt-in; defaults to ./mlruns and never connects to DagsHub."""
    if not enabled:
        return
    try:
        import mlflow
    except ImportError as e:
        raise RuntimeError('For --tracking install: pip install -e ".[tracking]"') from e
    uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("house-purchase-decision")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (float, int))})
        if (artifact_dir / "curves.png").exists():
            mlflow.log_artifact(str(artifact_dir / "curves.png"))
        mlflow.log_artifact(str(artifact_dir / "metrics.json"))
