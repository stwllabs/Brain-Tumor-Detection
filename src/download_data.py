"""
Download Brain Tumor MRI Dataset.

Dataset: Brain Tumor MRI Dataset (Figshare + SARTAJ + Br35H, dikurasi Chaki)
DOI: 10.21227/1jny-g144 (IEEE DataPort)
Mirror Kaggle: masoudnickparvar/brain-tumor-mri-dataset

Sebelum jalanin script ini:
1. Buat akun Kaggle (kaggle.com) kalau belum punya.
2. Ambil API token: Kaggle -> Account -> Create New API Token -> download kaggle.json
3. Set environment variable (lebih gampang daripada taruh file kaggle.json):
   export KAGGLE_USERNAME=<username_kamu>
   export KAGGLE_KEY=<key_dari_kaggle.json>
   (Windows: set KAGGLE_USERNAME=... / set KAGGLE_KEY=...)
"""

import os
import shutil
import kagglehub

DATASET_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    print(f"Downloading dataset: {DATASET_SLUG}")
    print("(Sumber: Chaki, J. Brain Tumor MRI Dataset. IEEE DataPort. "
          "DOI: 10.21227/1jny-g144)")

    cache_path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Downloaded to cache: {cache_path}")

    os.makedirs(TARGET_DIR, exist_ok=True)
    for item in os.listdir(cache_path):
        src = os.path.join(cache_path, item)
        dst = os.path.join(TARGET_DIR, item)
        if not os.path.exists(dst):
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    print(f"Dataset siap di: {os.path.abspath(TARGET_DIR)}")
    print("Struktur yang diharapkan: data/Training/<class>/*.jpg dan data/Testing/<class>/*.jpg")


if __name__ == "__main__":
    main()
