"""
Gabungkan semua *_summary.json (baseline, transfer, augmented, dst) jadi
satu tabel perbandingan -> langsung dipakai di bagian "Model Comparison" report.

Jalankan setelah minimal 2 model sudah di-training & di-summary-kan.
"""

import glob
import json
import os

import pandas as pd

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")


def main():
    summary_files = glob.glob(os.path.join(REPORT_DIR, "*_summary.json"))
    if not summary_files:
        print("Belum ada file *_summary.json. Training model dulu (train_baseline.py, dst).")
        return

    rows = []
    for f in summary_files:
        with open(f) as fp:
            rows.append(json.load(fp))

    df = pd.DataFrame(rows)
    df = df[["model_name", "n_params", "epochs_trained", "train_time_sec", "test_accuracy", "test_loss"]]
    df = df.sort_values("test_accuracy", ascending=False)

    print(df.to_string(index=False))

    out_path = os.path.join(REPORT_DIR, "model_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\nCatatan: akurasi tertinggi belum tentu paling 'suitable' untuk deployment --")
    print("bandingkan juga train_time_sec dan n_params (proxy computational cost) di report kamu.")


if __name__ == "__main__":
    main()
