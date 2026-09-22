"""
Eksperimen 1: Transfer learning pakai EfficientNetB0 (pretrained ImageNet).

Beda dari baseline:
- Pakai backbone pretrained, di-fine-tune, bukan CNN dari scratch
- Ini yang membuktikan pretrained model TIDAK dipakai sebagai black box:
  kita bandingkan langsung head-to-head vs baseline pakai metric & training curve yang sama

Jalankan: python src/train_transfer.py
Output: models/transfer_efficientnet.keras, report/transfer_history.csv,
        report/figures/transfer_training_curves.png, report/transfer_summary.json
"""

import json
import os
import time

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from preprocess import load_datasets

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
EPOCHS_HEAD = 10       # training head dulu, backbone dibekukan
EPOCHS_FINETUNE = 10   # lalu fine-tune sebagian backbone
NUM_CLASSES = 4


def build_transfer_model():
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # bekukan dulu untuk tahap 1

    inputs = tf.keras.Input(shape=(224, 224, 3))
    # EfficientNet punya preprocessing sendiri (beda dari rescaling 1/255 biasa),
    # input dari preprocess.py sudah 0-1, jadi kita scale ke 0-255 lalu preprocess resmi
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="transfer_efficientnet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def plot_curves(history_combined, out_path, finetune_start_epoch):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history_combined["loss"], label="train")
    axes[0].plot(history_combined["val_loss"], label="val")
    axes[0].axvline(finetune_start_epoch, color="gray", linestyle="--", label="fine-tune start")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history_combined["accuracy"], label="train")
    axes[1].plot(history_combined["val_accuracy"], label="val")
    axes[1].axvline(finetune_start_epoch, color="gray", linestyle="--", label="fine-tune start")
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

    model, base_model = build_transfer_model()
    model.summary()
    n_params = model.count_params()
    print(f"\nTotal parameters: {n_params:,}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    start = time.time()

    print("\n=== Tahap 1: training head (backbone dibekukan) ===")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks)

    print("\n=== Tahap 2: fine-tune (unfreeze bagian akhir backbone) ===")
    base_model.trainable = True
    # Bekukan layer awal, cuma fine-tune ~30% layer terakhir (lebih stabil, lebih cepat)
    freeze_until = int(len(base_model.layers) * 0.7)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # lr kecil untuk fine-tune
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, callbacks=callbacks)

    train_time = time.time() - start
    print(f"\nTraining time: {train_time:.1f}s ({train_time/60:.1f} min)")

    # Gabung history 2 tahap jadi 1 supaya bisa diplot & dibandingkan
    combined = {}
    for key in history1.history:
        combined[key] = history1.history[key] + history2.history[key]
    finetune_start_epoch = len(history1.history["loss"])

    model_path = os.path.join(MODEL_DIR, "transfer_efficientnet.keras")
    model.save(model_path)
    print(f"Saved model: {model_path}")

    hist_df = pd.DataFrame(combined)
    hist_path = os.path.join(REPORT_DIR, "transfer_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"Saved history: {hist_path}")

    plot_curves(combined, os.path.join(FIG_DIR, "transfer_training_curves.png"), finetune_start_epoch)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    summary = {
        "model_name": "transfer_efficientnet",
        "n_params": int(n_params),
        "train_time_sec": round(train_time, 1),
        "epochs_trained": len(combined["loss"]),
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
    }
    summary_path = os.path.join(REPORT_DIR, "transfer_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
