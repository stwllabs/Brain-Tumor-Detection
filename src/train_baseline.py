"""
Baseline CNN (dari scratch, tanpa transfer learning, tanpa augmentation).
Titik pembanding untuk semua eksperimen selanjutnya.

Jalankan: python src/train_baseline.py
Output:
- models/baseline_cnn.keras
- report/baseline_history.csv
- report/figures/baseline_training_curves.png
- accuracy di test set (dicetak ke terminal)
"""

import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from preprocess import load_datasets

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
EPOCHS = 20
NUM_CLASSES = 4


def build_baseline_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),

        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="baseline_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets()

    model = build_baseline_model()
    model.summary()

    # Jumlah parameter -> dicatat, dipakai buat tabel perbandingan model nanti
    n_params = model.count_params()
    print(f"\nTotal parameters: {n_params:,}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
    ]

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    train_time = time.time() - start
    print(f"\nTraining time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # Simpan model
    model_path = os.path.join(MODEL_DIR, "baseline_cnn.keras")
    model.save(model_path)
    print(f"Saved model: {model_path}")

    # Simpan history mentah (dipakai untuk analisis/perbandingan nanti)
    hist_df = pd.DataFrame(history.history)
    hist_path = os.path.join(REPORT_DIR, "baseline_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"Saved history: {hist_path}")

    plot_curves(history, os.path.join(FIG_DIR, "baseline_training_curves.png"))

    # Evaluasi di test set (dipegang terpisah sejak awal)
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # Simpan ringkasan singkat -> dipakai untuk tabel perbandingan antar eksperimen
    summary = {
        "model_name": "baseline_cnn",
        "n_params": int(n_params),
        "train_time_sec": round(train_time, 1),
        "epochs_trained": len(history.history["loss"]),
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
    }
    summary_path = os.path.join(REPORT_DIR, "baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
