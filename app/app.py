"""
Streamlit app: upload MRI otak -> prediksi kelas tumor.

Jalankan dari root folder project:
    streamlit run app/app.py

Catatan preprocessing: SEMUA model di project ini dilatih dengan input berskala 0-1
(lihat src/preprocess.py). Model transfer learning sudah punya layer Rescaling(255)
+ preprocess EfficientNet DI DALAM model, jadi app cukup memberi input 0-1 untuk semua model.
"""

import glob
import os
import time

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # urutan sesuai preprocess.py
IMG_SIZE = (224, 224)

st.set_page_config(page_title="NeuroScan | MRI Classifier", page_icon="🧠", layout="wide")


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0  # skala 0-1, sama seperti saat training
    return np.expand_dims(arr, axis=0)


def main():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root {
            --ink: #172b2a;
            --muted: #667875;
            --teal: #087e75;
            --mint: #dff4ed;
            --line: #dce8e3;
            --paper: #f7faf8;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 94% 0%, #e5f4ed 0, transparent 28rem), var(--paper); }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1240px; padding: 2.2rem 2.5rem 4rem; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink) !important; letter-spacing: 0 !important; }
        [data-testid="stSidebar"] { background: #edf5f1; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] h2 { font-size: 1.05rem; }
        [data-testid="stFileUploader"] section { background: #fff; border: 1px dashed #9dbdb2; border-radius: 10px; }
        [data-testid="stFileUploader"] section:hover { border-color: var(--teal); }
        [data-testid="stImage"] img { border-radius: 8px; }
        [data-testid="stProgress"] > div > div { background: var(--teal); }
        [data-testid="stProgress"] > div { background: #e4eeea; }
        .hero { padding: 1rem 0 1.65rem; border-bottom: 1px solid var(--line); margin-bottom: 1.65rem; }
        .eyebrow { color: var(--teal); text-transform: uppercase; font-size: .72rem; font-weight: 700; letter-spacing: .11em; }
        .hero-title { font: 800 2.45rem/1.12 'Manrope', sans-serif; color: var(--ink); margin: .5rem 0 .55rem; }
        .hero-copy { color: var(--muted); max-width: 650px; font-size: 1rem; margin: 0; }
        .disclaimer { border-left: 3px solid #d39444; background: #fff7e9; color: #765321; padding: .75rem 1rem; border-radius: 0 6px 6px 0; font-size: .82rem; margin-top: 1.25rem; }
        .section-kicker { color: var(--muted); font-size: .72rem; text-transform: uppercase; font-weight: 700; letter-spacing: .1em; margin: .15rem 0 .7rem; }
        .result-name { color: var(--ink); font: 800 1.8rem/1.15 'Manrope', sans-serif; text-transform: capitalize; margin: .4rem 0 .2rem; overflow-wrap: anywhere; }
        .result-meta { color: var(--muted); font-size: .86rem; }
        .confidence { color: var(--teal); font: 800 2.5rem/1 'Manrope', sans-serif; margin: .7rem 0 .25rem; }
        .empty-state { min-height: 245px; display: grid; place-content: center; text-align: center; border: 1px solid var(--line); background: rgba(255,255,255,.72); border-radius: 8px; padding: 2rem; }
        .empty-icon { font-size: 2rem; margin-bottom: .5rem; }
        .empty-title { font: 700 1.05rem 'Manrope', sans-serif; color: var(--ink); }
        .empty-copy { color: var(--muted); font-size: .86rem; margin-top: .35rem; }
        @media (max-width: 700px) {
            .block-container { padding: 1.3rem 1rem 3rem; }
            .hero-title { font-size: 1.9rem; }
        }
        </style>
        <header class="hero">
            <div class="eyebrow">COMP6826001 · Prototipe klasifikasi MRI</div>
            <div class="hero-title">NeuroScan</div>
            <p class="hero-copy">Eksplorasi klasifikasi citra MRI otak dengan model deep learning.</p>
            <div class="disclaimer"><strong>Catatan penting:</strong> Ini prototipe akademik, bukan alat diagnosis medis. Jangan gunakan hasilnya untuk keputusan klinis.</div>
        </header>
        """,
        unsafe_allow_html=True,
    )

    model_files = glob.glob(os.path.join(MODEL_DIR, "*.keras"))
    if not model_files:
        st.error("Belum ada model di folder `models/`. Training dulu (misal `python src/train_baseline.py`).")
        return

    model_options = {os.path.basename(f): f for f in model_files}
    with st.sidebar:
        st.markdown("## Pengaturan")
        st.caption("Pilih model untuk menjalankan inferensi.")
        selected_name = st.selectbox("Model", list(model_options.keys()))
        st.divider()
        st.caption("4 kelas · Input 224 × 224 px")
    model_path = model_options[selected_name]

    upload_col, result_col = st.columns([1, 1.08], gap="large")
    with upload_col:
        st.markdown('<div class="section-kicker">01 / Gambar MRI</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Unggah gambar MRI otak", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption=uploaded_file.name, use_container_width=True)
        else:
            st.markdown(
                '<div class="empty-state"><div class="empty-icon">◉</div><div class="empty-title">Siap menerima gambar</div><div class="empty-copy">Pilih file JPG atau PNG untuk memulai analisis.</div></div>',
                unsafe_allow_html=True,
            )

    with result_col:
        st.markdown('<div class="section-kicker">02 / Hasil analisis</div>', unsafe_allow_html=True)
        if uploaded_file is None:
            st.markdown(
                '<div class="empty-state"><div class="empty-icon">⌁</div><div class="empty-title">Hasil akan tampil di sini</div><div class="empty-copy">Unggah citra MRI untuk melihat prediksi dan distribusi probabilitas.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Menganalisis citra..."):
                model = load_model(model_path)
                x = preprocess_image(img)
                t0 = time.time()
                preds = model.predict(x, verbose=0)[0]
                latency_ms = (time.time() - t0) * 1000

            pred_idx = int(np.argmax(preds))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(preds[pred_idx])
            st.markdown(f'<div class="result-meta">KELAS TERPREDIKSI</div><div class="result-name">{pred_class}</div><div class="confidence">{confidence:.1%}</div><div class="result-meta">Keyakinan model · Inferensi {latency_ms:.0f} ms</div>', unsafe_allow_html=True)
            st.divider()
            st.markdown("**Distribusi probabilitas**")
            for cls, prob in sorted(zip(CLASS_NAMES, preds), key=lambda x: -x[1]):
                label_col, value_col = st.columns([3, 1])
                label_col.markdown(f"`{cls}`")
                value_col.markdown(f"**{float(prob):.1%}**")
                st.progress(float(prob))

            if confidence < 0.6:
                st.warning("Keyakinan model rendah. Hasil perlu ditinjau secara manual dan tidak boleh dijadikan kesimpulan.")
            elif pred_class == "notumor":
                st.info("Prediksi 'notumor' tidak menjamin pasien bebas tumor. Konfirmasi hasil dengan tenaga medis.")


if __name__ == "__main__":
    main()