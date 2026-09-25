"""
Streamlit app: upload MRI otak -> prediksi kelas tumor.

Jalankan dari root folder project:
    streamlit run app/app.py
"""

import glob
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # urutan sesuai preprocess.py
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_image(img: Image.Image, model_name: str):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")

    if "efficientnet" in model_name.lower() or "transfer" in model_name.lower():
        # Model transfer learning punya preprocessing resmi EfficientNet
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    else:
        # Baseline CNN pakai rescaling 0-1 biasa (sesuai preprocess.py)
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)


def main():
    st.title("🧠 Brain Tumor Detection")
    st.caption(
        "Prototipe tugas kuliah COMP6826001 — BUKAN alat diagnosis medis. "
        "Hasil prediksi tidak boleh dipakai untuk keputusan klinis nyata."
    )

    # Pilih model yang tersedia di folder models/
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.keras"))
    if not model_files:
        st.error("Belum ada model di folder `models/`. Training dulu (misal `python src/train_baseline.py`).")
        return

    model_options = {os.path.basename(f): f for f in model_files}
    selected_name = st.selectbox("Pilih model", list(model_options.keys()))
    model_path = model_options[selected_name]

    uploaded_file = st.file_uploader("Upload gambar MRI otak (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diupload", width=300)

        with st.spinner("Menganalisis..."):
            model = load_model(model_path)
            x = preprocess_image(img, selected_name)
            preds = model.predict(x, verbose=0)[0]

        pred_idx = int(np.argmax(preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx])

        st.subheader(f"Prediksi: **{pred_class}**")
        st.write(f"Confidence: {confidence:.1%}")

        st.write("Probabilitas semua kelas:")
        for cls, prob in sorted(zip(CLASS_NAMES, preds), key=lambda x: -x[1]):
            st.write(f"- {cls}: {prob:.1%}")
            st.progress(float(prob))

        if pred_class != "notumor" and confidence < 0.6:
            st.warning(
                "Confidence rendah — kalau ini aplikasi nyata, kasus seperti ini idealnya "
                "tetap dirujuk ke radiolog untuk review manual, bukan diandalkan langsung."
            )


if __name__ == "__main__":
    main()
