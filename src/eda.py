"""
Exploratory Data Analysis dasar untuk Brain Tumor MRI Dataset.

Menghasilkan:
- Jumlah gambar per kelas (Training & Testing)
- Distribusi ukuran gambar (cek apakah seragam)
- Grid contoh gambar per kelas
- Ringkasan disimpan ke report/eda_summary.csv dan report/figures/

Jalankan setelah src/download_data.py selesai.
"""

import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
CLASSES = ["notumor", "glioma", "meningioma", "pituitary"]
SPLITS = ["Training", "Testing"]


def count_per_class():
    rows = []
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"[!] Folder tidak ditemukan: {split_dir} — cek hasil download_data.py")
            continue
        for cls in CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                n = len([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                rows.append({"split": split, "class": cls, "count": n})
    return pd.DataFrame(rows)


def plot_class_distribution(df):
    os.makedirs(FIG_DIR, exist_ok=True)
    pivot = df.pivot(index="class", columns="split", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(7, 4))
    ax.set_ylabel("Jumlah gambar")
    ax.set_title("Distribusi kelas: Brain Tumor MRI Dataset")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "class_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def check_image_sizes(sample_per_class=30):
    sizes = defaultdict(list)
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, "Training", cls)
        if not os.path.isdir(cls_dir):
            continue
        files = os.listdir(cls_dir)
        sample = random.sample(files, min(sample_per_class, len(files)))
        for f in sample:
            with Image.open(os.path.join(cls_dir, f)) as img:
                sizes[cls].append(img.size)

    print("\n--- Ukuran gambar (sampel) ---")
    for cls, s in sizes.items():
        widths = [w for w, h in s]
        heights = [h for w, h in s]
        if widths:
            print(f"{cls}: width {min(widths)}-{max(widths)}, height {min(heights)}-{max(heights)}")
    return sizes


def plot_sample_grid():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(len(CLASSES), 4, figsize=(10, 10))
    for i, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATA_DIR, "Training", cls)
        if not os.path.isdir(cls_dir):
            continue
        files = random.sample(os.listdir(cls_dir), min(4, len(os.listdir(cls_dir))))
        for j, f in enumerate(files):
            img = Image.open(os.path.join(cls_dir, f))
            axes[i, j].imshow(img, cmap="gray")
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(cls, fontsize=10)
        axes[i, 0].set_title(cls, fontsize=10, loc="left")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "sample_grid.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = count_per_class()
    if df.empty:
        print("Tidak ada data ditemukan. Jalankan src/download_data.py dulu.")
        return

    print("\n--- Jumlah gambar per kelas ---")
    print(df.pivot(index="class", columns="split", values="count").fillna(0))

    df.to_csv(os.path.join(REPORT_DIR, "eda_summary.csv"), index=False)
    print(f"\nSaved summary: {os.path.join(REPORT_DIR, 'eda_summary.csv')}")

    plot_class_distribution(df)
    check_image_sizes()
    plot_sample_grid()

    print("\nEDA selesai. Cek report/figures/ untuk grafik dan report/eda_summary.csv untuk angka.")


if __name__ == "__main__":
    main()
