# Brain Tumor Detection — Final Project COMP6826001

Deep learning decision-support tool untuk klasifikasi MRI otak: **no tumor / glioma / meningioma / pituitary**.

## Struktur Project
```
brain-tumor-detection/
├── data/                  # dataset (tidak di-commit ke git, lihat .gitignore)
├── notebooks/             # eksplorasi & eksperimen
├── src/
│   ├── download_data.py   # download dataset dari Kaggle
│   └── eda.py              # exploratory data analysis
├── app/                    # Streamlit app (dibuat belakangan)
├── report/                 # technical report, AI usage log
├── requirements.txt
└── README.md
```

## Dataset
**Brain Tumor MRI Dataset** (gabungan Figshare + SARTAJ + Br35H, dikurasi Chaki)
- DOI: [10.21227/1jny-g144](https://doi.org/10.21227/1jny-g144) (IEEE DataPort)
- Mirror Kaggle: `masoudnickparvar/brain-tumor-mri-dataset`
- 7.023 gambar MRI, 4 kelas: glioma, meningioma, pituitary, notumor
- Sudah terbagi folder `Training/` dan `Testing/` di sumber aslinya

## Cara Mulai
```bash
pip install -r requirements.txt
python src/download_data.py     # download & extract dataset ke data/
python src/eda.py               # EDA: distribusi kelas, contoh gambar, ukuran gambar
```

## Status
- [x] Setup repo & struktur folder
- [x] Script download dataset
- [x] EDA dasar
- [ ] Preprocessing pipeline
- [ ] Baseline CNN
- [ ] Eksperimen (transfer learning, augmentation, dll)
- [ ] Evaluasi & analisis
- [ ] Streamlit app
- [ ] Technical report
