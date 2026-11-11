"""TF/Keras models mirroring the original research architectures."""
from __future__ import annotations

FIXED_OPTIMIZED_PARAMS = {
    "n_layers": 4, "units_0": 224, "units_1": 64,
    "units_2": 48, "units_3": 48, "dropout": .15,
    "l2_regularization": 5.4714502169763595e-05,
    "batch_norm": False, "learning_rate": 0.0019666963651124845,
}


def build_model(input_dim: int, architecture: str = "original", params: dict | None = None):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    model = keras.Sequential(name=f"house_purchase_{architecture}")
    model.add(layers.Input(shape=(input_dim,)))
    if architecture == "original":
        for i, units in enumerate((128, 64, 32)):
            model.add(layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))
            if i == 0:
                model.add(layers.BatchNormalization())
            model.add(layers.Dropout(.2))
        lr = .001
    elif architecture in {"optimized", "tuned"}:
        hp = params or FIXED_OPTIMIZED_PARAMS
        for i in range(int(hp["n_layers"])):
            model.add(layers.Dense(int(hp[f"units_{i}"]), kernel_regularizer=regularizers.l2(float(hp["l2_regularization"]))))
            model.add(layers.LeakyReLU(negative_slope=.1))
            if hp["batch_norm"]:
                model.add(layers.BatchNormalization())
            model.add(layers.Dropout(float(hp["dropout"])))
        lr = float(hp["learning_rate"])
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.AUC(name="roc_auc"),
                 keras.metrics.AUC(name="pr_auc", curve="PR")],
    )
    return model


def fit_callbacks():
    from tensorflow import keras
    return [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=4, min_lr=1e-6),
    ]
