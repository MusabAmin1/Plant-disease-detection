import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_mobilenetv2.keras")

@st.cache_data
def load_classes():
    with open("class_names.json") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

def clean(name):
    return name.replace("___", " - ").replace("__", " ").replace("_", " ")

def predict(img, top_k=3):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)[None, ...]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    probs = model.predict(arr, verbose=0)[0]
    idx = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in idx]

st.title("🌿 Plant Disease Detection")
st.write("Upload a clear photo of a single leaf.")
st.info("Supported plants: Pepper, Potato, Tomato")

file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)

    with st.spinner("Analyzing..."):
        results = predict(img)

    label, conf = results[0]
    if conf < 0.50:
        st.warning(f"Low confidence: best guess is **{clean(label)}** ({conf:.0%}). "
                   "Please try a clearer, closer photo of one leaf.")
    else:
        st.success(f"**{clean(label)}** ({conf:.1%})")

    st.subheader("Top predictions")
    for name, p in results:
        st.write(f"{clean(name)}: {p:.1%}")
        st.progress(p)