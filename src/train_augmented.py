"""
Eksperimen 2: Baseline CNN (arsitektur sama persis dengan train_baseline.py)
tapi DENGAN data augmentation. Dibandingkan head-to-head ke baseline untuk
lihat pengaruh augmentation terhadap overfitting & generalisasi ke test set.

Jalankan: python src/train_augmented.py
"""

import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from preprocess import load_datasets, get_augmentation_layer
from train_baseline import build_baseline_model, plot_curves  # reuse arsitektur & plot yang sama

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
EPOCHS = 20


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets()

    # Terapkan augmentation HANYA ke train_ds (val & test harus tetap bersih/asli)
    augmentation = get_augmentation_layer()
    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))

    model = build_baseline_model()  # arsitektur identik dengan baseline
    model._name = "baseline_cnn_augmented"
    n_params = model.count_params()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    train_time = time.time() - start
    print(f"\nTraining time: {train_time:.1f}s ({train_time/60:.1f} min)")

    model_path = os.path.join(MODEL_DIR, "baseline_augmented.keras")
    model.save(model_path)
    print(f"Saved model: {model_path}")

    hist_df = pd.DataFrame(history.history)
    hist_path = os.path.join(REPORT_DIR, "augmented_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"Saved history: {hist_path}")

    plot_curves(history, os.path.join(FIG_DIR, "augmented_training_curves.png"))

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    summary = {
        "model_name": "baseline_augmented",
        "n_params": int(n_params),
        "train_time_sec": round(train_time, 1),
        "epochs_trained": len(history.history["loss"]),
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
    }
    summary_path = os.path.join(REPORT_DIR, "augmented_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
