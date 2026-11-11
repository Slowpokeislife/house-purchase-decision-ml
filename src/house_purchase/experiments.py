"""Train, optimize, evaluate and persist reproducible artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import random
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

from .config import DEFAULT_DATA, SEED
from .data import prepare_data, training_input_schema
from .model import FIXED_OPTIMIZED_PARAMS, build_model, fit_callbacks
from .report import metrics_at_threshold, maybe_log_mlflow, save_learning_curves, save_metrics


def _load(path, seed):
    return prepare_data(Path(path), seed)


def _store_common(out, processor, splits, model_kind, seed):
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(processor, out / "preprocessor.joblib")
    (out / "input_schema.json").write_text(json.dumps(training_input_schema(splits.X_train), indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({"model_kind": model_kind, "seed": seed,
        "train_rows": len(splits.y_train), "val_rows": len(splits.y_val), "test_rows": len(splits.y_test),
        "test_evaluated_after_model_selection": True}, indent=2), encoding="utf-8")


def _predict_nn(model, array):
    return model.predict(array, verbose=0).ravel()


def _weights(y):
    labels = np.unique(y)
    vals = compute_class_weight(class_weight="balanced", classes=labels, y=np.asarray(y))
    return {int(label): float(weight) for label, weight in zip(labels, vals)}


def run_baseline(data=DEFAULT_DATA, out=Path("artifacts/baseline"), seed=SEED, tracking=False):
    splits, prep, train, val, test = _load(data, seed)
    model = LogisticRegression(class_weight="balanced", max_iter=1500, random_state=seed)
    model.fit(train, splits.y_train)
    val_metrics = metrics_at_threshold(splits.y_val, model.predict_proba(val)[:, 1])
    test_metrics = metrics_at_threshold(splits.y_test, model.predict_proba(test)[:, 1])
    out = Path(out)
    _store_common(out, prep, splits, "baseline", seed)
    joblib.dump(model, out / "model.joblib")
    save_metrics(out / "validation_metrics.json", val_metrics)
    save_metrics(out / "metrics.json", test_metrics)
    maybe_log_mlflow("logistic_baseline", out, test_metrics, {"model": "logistic_regression"}, tracking)
    return test_metrics


def _train_nn(prep, splits, train, val, test, model, out, seed, epochs, batch_size, tracking, params):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    history = model.fit(train, splits.y_train.to_numpy(),
        validation_data=(val, splits.y_val.to_numpy()), epochs=epochs, batch_size=batch_size,
        class_weight=_weights(splits.y_train), callbacks=fit_callbacks(), verbose=2)
    val_metrics = metrics_at_threshold(splits.y_val, _predict_nn(model, val))
    # Only NOW consult test set: never use it inside Optuna or Keras callbacks.
    test_metrics = metrics_at_threshold(splits.y_test, _predict_nn(model, test))
    out = Path(out)
    _store_common(out, prep, splits, "keras", seed)
    model.save(out / "model.keras")
    save_metrics(out / "validation_metrics.json", val_metrics)
    save_metrics(out / "metrics.json", test_metrics)
    save_learning_curves(history.history, out / "curves.png")
    maybe_log_mlflow("keras_model", out, test_metrics, params, tracking)
    return test_metrics


def run_training(architecture="optimized", data=DEFAULT_DATA, out=None,
                 seed=SEED, epochs=100, batch_size=64, tracking=False):
    if architecture not in {"original", "optimized"}:
        raise ValueError("Valid architectures: original, optimized")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    splits, prep, train, val, test = _load(data, seed)
    model = build_model(train.shape[1], architecture)
    out = Path(out or f"artifacts/{architecture}")
    return _train_nn(prep, splits, train, val, test, model, out,
                     seed, epochs, batch_size, tracking,
                     {"architecture": architecture, "batch_size": batch_size,
                      "epochs_max": epochs, **(FIXED_OPTIMIZED_PARAMS if architecture=="optimized" else {})})


def run_tuning(data=DEFAULT_DATA, out=Path("artifacts/tuned"), seed=SEED,
               trials=20, trial_epochs=25, final_epochs=100, tracking=False):
    import tensorflow as tf
    import optuna
    splits, prep, train, val, test = _load(data, seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)

    def objective(trial):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed + trial.number)
        params = {
            "n_layers": trial.suggest_int("n_layers", 2, 4),
            "dropout": trial.suggest_float("dropout", .1, .4, step=.05),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-5, 1e-3, log=True),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }
        prev = trial.suggest_int("units_0", 64, 256, step=32)
        params["units_0"] = prev
        for i in range(1, params["n_layers"]):
            # Keep same search logic as the original notebook: non-increasing widths.
            prev = trial.suggest_int(f"units_{i}", 16, prev, step=16)
            params[f"units_{i}"] = prev
        model = build_model(train.shape[1], "tuned", params)
        model.fit(train, splits.y_train.to_numpy(),
            validation_data=(val, splits.y_val.to_numpy()), epochs=trial_epochs,
            batch_size=128, class_weight=_weights(splits.y_train),
            callbacks=fit_callbacks(), verbose=0)
        from sklearn.metrics import average_precision_score
        score = float(average_precision_score(splits.y_val, _predict_nn(model, val)))
        trial.set_user_attr("validation_average_precision", score)
        del model
        return score

    # Local SQLite study, stored outside source control. Seed reproducible trials.
    storage = "sqlite:///" + str((out / "optuna.db").resolve()).replace(chr(92), "/")
    study = optuna.create_study(study_name="purchase_decision", storage=storage,
                                load_if_exists=True, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    params = dict(best.params)
    (out / "best_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (out / "tuning_summary.json").write_text(json.dumps({
        "criterion": "validation_average_precision", "best_value": best.value,
        "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "test_was_not_used_for_search": True
    }, indent=2), encoding="utf-8")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(train.shape[1], "tuned", params)
    return _train_nn(prep, splits, train, val, test, model, out, seed,
                     final_epochs, 64, tracking, params)
