"""
Evaluasi generik: confusion matrix + classification report + cek pola error.
Reusable untuk model manapun (baseline, transfer learning, dst).

Jalankan: python src/evaluate.py --model models/baseline_cnn.keras --name baseline
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import load_datasets

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")
FIG_DIR = os.path.join(REPORT_DIR, "figures")


def evaluate_model(model_path, run_name):
    os.makedirs(FIG_DIR, exist_ok=True)

    _, _, test_ds, class_names = load_datasets()
    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Classification report (precision/recall/F1 per kelas)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    report_path = os.path.join(REPORT_DIR, f"{run_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {run_name}")
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, f"{run_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")

    # Ringkasan error per kelas -> kelas mana yang paling sering ketuker jadi apa
    print("\n--- Pola kesalahan terbesar per kelas ---")
    for i, cls in enumerate(class_names):
        row = cm[i].copy()
        n_total = row.sum()
        row[i] = 0  # exclude correct predictions
        if row.sum() > 0:
            worst_idx = row.argmax()
            print(f"{cls}: {row[worst_idx]}/{n_total} salah diprediksi sebagai '{class_names[worst_idx]}'")

    return cm, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path ke file .keras")
    parser.add_argument("--name", required=True, help="Nama run, dipakai untuk nama file output")
    args = parser.parse_args()

    evaluate_model(args.model, args.name)
