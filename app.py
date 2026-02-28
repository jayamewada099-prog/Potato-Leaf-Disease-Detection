import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import joblib
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🌾 AgroVision AI",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_all_models():
    cnn_model = load_model("models/potato_cnn_model.keras")
    pca_model = joblib.load("models/pca_model.pkl")
    lr_model = joblib.load("models/lr_gwo_model.pkl")
    return cnn_model, pca_model, lr_model

cnn_model, pca_model, lr_model = load_all_models()

class_names = ["Early_Blight", "Healthy", "Late_Blight"]

# ---------------- UI ----------------
st.title("🌾 AgroVision AI")
st.subheader("Potato Leaf Disease Detection System (CNN + PCA + GWO)")
st.write("Upload a potato leaf image to detect disease using optimized AI pipeline.")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Detect Disease"):

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        # -------- IMAGE PREPROCESSING --------
        img = image.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # -------- CNN FEATURE EXTRACTION --------
        features = cnn_model.predict(img_array)
        features = features.reshape(1, -1)

        # -------- PCA TRANSFORM --------
        pca_features = pca_model.transform(features)

        # -------- GWO-OPTIMIZED LR PREDICTION --------
        prediction_index = lr_model.predict(pca_features)[0]
        confidence = np.max(lr_model.predict_proba(pca_features)) * 100

        prediction = class_names[prediction_index]

        # -------- SHOW RESULTS --------
        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {round(confidence,2)}%")

        # -------- PIE CHART --------
        fig, ax = plt.subplots()
        ax.pie(
            [confidence, 100 - confidence],
            labels=["Confidence", "Remaining"],
            autopct='%1.1f%%'
        )
        st.pyplot(fig)