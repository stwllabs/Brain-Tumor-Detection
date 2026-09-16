"""
Cek manual orientasi MRI (axial / coronal / sagittal) per kelas.

Cara pakai:
1. Jalankan script ini -> hasilnya grid gambar per kelas disimpan ke report/figures/orientation_check_<kelas>.png
2. Buka tiap file, itung manual berapa yang axial vs coronal/sagittal
   - Axial: potongan horizontal (dari atas), biasanya bentuk oval/bulat simetris kiri-kanan,
     terlihat seperti "irisan donat", mata kadang kelihatan kalau potongan rendah
   - Coronal: potongan depan (dari muka), bentuk lebih vertikal, terlihat seperti wajah dibelah dua
   - Sagittal: potongan samping (dari telinga), terlihat profil (hidung, batang otak kelihatan jelas)
3. Catat hasil hitungan manual kamu, nanti dipakai untuk analisis dataset bias di report
"""

import os
import random

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
CLASSES = ["notumor", "glioma", "meningioma", "pituitary"]
N_SAMPLES = 20  # per kelas


def save_grid_for_class(cls, n=N_SAMPLES, cols=5):
    cls_dir = os.path.join(DATA_DIR, "Training", cls)
    files = os.listdir(cls_dir)
    sample = random.sample(files, min(n, len(files)))
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.flatten()
    for i, f in enumerate(sample):
        img = Image.open(os.path.join(cls_dir, f))
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(str(i + 1), fontsize=9)
        axes[i].axis("off")
    for j in range(len(sample), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Kelas: {cls} (n={len(sample)}) — beri nomor tiap gambar saat menghitung", fontsize=12)
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, f"orientation_check_{cls}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}  -> buka & itung manual axial vs coronal/sagittal")


def main():
    for cls in CLASSES:
        save_grid_for_class(cls)
    print("\nSelesai. Buka semua file report/figures/orientation_check_*.png,")
    print("itung manual per kelas, lalu catat hasilnya (misal: notumor 18/20 axial, glioma 2/20 axial, dst).")


if __name__ == "__main__":
    main()
