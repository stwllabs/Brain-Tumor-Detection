"""Streamlit app for brain MRI image classification.

Run from the project root with:
    streamlit run app/app.py

All models in this project were trained with inputs scaled to 0-1 (see
src/preprocess.py). The transfer-learning model includes its own Rescaling(255)
and EfficientNet preprocessing layers, so the app supplies 0-1 inputs to every model.
"""

import glob
import json
import os
import time

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # Order matches preprocess.py.
DISPLAY_NAMES = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumor", "pituitary": "Pituitary"}
IMG_SIZE = (224, 224)

st.set_page_config(page_title="NeuroScan | MRI Classifier", page_icon="🧠", layout="wide")


@st.cache_resource
def load_model(model_path):
    if os.path.isdir(model_path):
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as config_file:
            model_config = json.load(config_file)
        model = tf.keras.models.model_from_json(json.dumps(model_config))
        model.load_weights(os.path.join(model_path, "model.weights.h5"))
        return model
    return tf.keras.models.load_model(model_path)


def preprocess_image(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0  # Scale to 0-1, matching training.
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
            <div class="eyebrow">MRI CLASSIFICATION PROTOTYPE</div>
            <div class="hero-title">NeuroScan</div>
            <p class="hero-copy">Explore brain MRI image classification with deep learning models.</p>
            <div class="disclaimer"><strong>Important:</strong> This is an academic prototype, not a medical diagnostic tool. Do not use its results to make clinical decisions.</div>
        </header>
        """,
        unsafe_allow_html=True,
    )

    model_files = glob.glob(os.path.join(MODEL_DIR, "*.keras"))
    if not model_files:
        st.error("No models were found in the `models/` folder. Train a model first, for example with `python src/train_baseline.py`.")
        return

    model_options = {os.path.basename(f): f for f in model_files}
    with st.sidebar:
        st.markdown("## Settings")
        st.caption("Choose a model for inference.")
        selected_name = st.selectbox("Model", list(model_options.keys()))
        st.divider()
        st.caption("4 classes · 224 × 224 px input")
    model_path = model_options[selected_name]

    upload_col, result_col = st.columns([1, 1.08], gap="large")
    with upload_col:
        st.markdown('<div class="section-kicker">01 / MRI IMAGE</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption=uploaded_file.name, use_container_width=True)
        else:
            st.markdown(
                '<div class="empty-state"><div class="empty-icon">◉</div><div class="empty-title">Ready for an image</div><div class="empty-copy">Choose a JPG or PNG file to begin analysis.</div></div>',
                unsafe_allow_html=True,
            )

    with result_col:
        st.markdown('<div class="section-kicker">02 / ANALYSIS</div>', unsafe_allow_html=True)
        if uploaded_file is None:
            st.markdown(
                '<div class="empty-state"><div class="empty-icon">⌁</div><div class="empty-title">Results will appear here</div><div class="empty-copy">Upload an MRI image to view the prediction and probability distribution.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Analyzing image..."):
                model = load_model(model_path)
                x = preprocess_image(img)
                t0 = time.time()
                preds = model.predict(x, verbose=0)[0]
                latency_ms = (time.time() - t0) * 1000

            pred_idx = int(np.argmax(preds))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(preds[pred_idx])
            st.markdown(f'<div class="result-meta">PREDICTED CLASS</div><div class="result-name">{DISPLAY_NAMES[pred_class]}</div><div class="confidence">{confidence:.1%}</div><div class="result-meta">Model confidence · Inference {latency_ms:.0f} ms</div>', unsafe_allow_html=True)
            st.divider()
            st.markdown("**Probability distribution**")
            for cls, prob in sorted(zip(CLASS_NAMES, preds), key=lambda x: -x[1]):
                label_col, value_col = st.columns([3, 1])
                label_col.markdown(f"`{DISPLAY_NAMES[cls]}`")
                value_col.markdown(f"**{float(prob):.1%}**")
                st.progress(float(prob))

            if confidence < 0.6:
                st.warning("Model confidence is low. Review this result manually; it should not be treated as a conclusion.")
            elif pred_class == "notumor":
                st.info("A 'No tumor' prediction does not guarantee that a patient is tumor-free. Confirm findings with a medical professional.")


if __name__ == "__main__":
    main()