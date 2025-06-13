import streamlit as st
from PIL import Image
import numpy as np

st.title("🌊 Marine Eco-AI: Oil Spill Detector")

# Dummy Prediction Placeholder (since model isn't uploaded yet)
def predict(image):
    return "⚠️ Oil Spill Detected" if np.random.rand() > 0.5 else "✅ Clean Ocean Surface"

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    result = predict(image)
    st.subheader("Prediction:")
    st.write(result)
