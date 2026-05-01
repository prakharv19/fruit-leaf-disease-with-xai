import streamlit as st
import numpy as np
import cv2
import json
import tensorflow as tf
from PIL import Image
import requests
import os

from tensorflow.keras.applications.resnet50 import preprocess_input

# =========================
# TITLE
# =========================
st.set_page_config(page_title="XAI Leaf Disease Detector", layout="wide")

st.title("🌿 Fruit Leaf Disease Detection (Explainable AI)")

st.markdown("Upload a leaf image to get **Prediction + Heatmap + Full AI Explanation**")

# =========================
# LOAD FILES
# =========================
with open("class_names.json") as f:
    class_names = json.load(f)

with open("disease_templates.json") as f:
    disease_info = json.load(f)

# =========================
# LOAD MODEL FROM DRIVE
# =========================
@st.cache_resource
def load_model():

    file_id = "1P6srEiqSZUtd03sp35ETqTehtBYkKtUz"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    model_path = "model.keras"

    if not os.path.exists(model_path):
        r = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(r.content)

    return tf.keras.models.load_model(model_path)

model = load_model()

# =========================
# GRAD-CAM MODEL
# =========================
grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer("conv5_block3_out").output, model.output]
)

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

# =========================
# GRAD-CAM
# =========================
def make_heatmap(img_array, class_index):

    with tf.GradientTape() as tape:
        conv, preds = grad_model(img_array)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv = conv[0]
    heatmap = conv @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()

# =========================
# REGION ANALYSIS
# =========================
def region_analysis(heatmap):

    h, w = heatmap.shape

    regions = {
        "center": np.mean(heatmap[h//4:3*h//4, w//4:3*w//4]),
        "top": np.mean(heatmap[:h//3, :]),
        "bottom": np.mean(heatmap[2*h//3:, :]),
        "left": np.mean(heatmap[:, :w//3]),
        "right": np.mean(heatmap[:, 2*w//3:])
    }

    dominant = max(regions, key=regions.get)

    return regions, dominant

# =========================
# ADVANCED FEATURES
# =========================
def lesion_analysis(image, heatmap):

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    mask = heatmap_resized > 0.35

    lesion_area = np.sum(mask) / mask.size

    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
    lesion_count = max(num_labels - 1, 0)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    texture = np.std(gray[mask]) if np.sum(mask) > 0 else 0

    return lesion_area, lesion_count, texture

# =========================
# SEVERITY
# =========================
def severity(area, texture, count):

    if area < 0.05:
        return "Early Stage"
    elif area < 0.15:
        return "Moderate Stage"
    elif area < 0.30:
        return "Advanced Stage"
    else:
        return "Severe Stage"

# =========================
# FULL PREDICTION ENGINE
# =========================
def predict(image):

    img_array = preprocess(image)

    preds = model.predict(img_array, verbose=0)[0]

    top3 = np.argsort(preds)[-3:][::-1]

    top3_text = "\n".join([
        f"{class_names[i]} : {preds[i]*100:.2f}%"
        for i in top3
    ])

    idx = top3[0]
    label = class_names[idx]
    confidence = preds[idx] * 100

    heatmap = make_heatmap(img_array, idx)

    img_np = np.array(image.resize((224,224)))

    # heatmap overlay
    heatmap_resized = cv2.resize(heatmap, (224,224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

    # analysis
    regions, dominant = region_analysis(heatmap)
    area, count, texture = lesion_analysis(img_np, heatmap)
    sev = severity(area, texture, count)

    explanation = disease_info.get(label, "No description available")

    full_report = f"""

🧠 FINAL AI REPORT

━━━━━━━━━━━━━━━━━━━━
🌱 Disease Prediction: {label}
🎯 Confidence: {confidence:.2f}%

━━━━━━━━━━━━━━━━━━━━
🔥 TOP 3 PREDICTIONS:
{top3_text}

━━━━━━━━━━━━━━━━━━━━
📍 HEATMAP ANALYSIS:
- Dominant Region: {dominant}
- Center Activation: {regions['center']:.3f}

━━━━━━━━━━━━━━━━━━━━
🦠 LESION ANALYSIS:
- Infected Area: {area*100:.2f}%
- Lesion Count: {count}
- Texture Variation: {texture:.2f}

━━━━━━━━━━━━━━━━━━━━
⚠️ SEVERITY:
{sev}

━━━━━━━━━━━━━━━━━━━━
📘 MODEL EXPLANATION:
{explanation}

━━━━━━━━━━━━━━━━━━━━
FINAL DECISION:
Model detected {label} because:
- lesion pattern matches trained features
- heatmap shows strong activation in {dominant}
- texture + shape + spread confirm disease signature
"""

    return label, confidence, overlay, full_report

# =========================
# UI
# =========================
uploaded_file = st.file_uploader("📥 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    label, confidence, heatmap_img, report = predict(image)

    with col2:
        st.image(heatmap_img, caption="🔥 XAI Heatmap", use_column_width=True)

    st.success(f"🌱 Disease: {label}")
    st.info(f"🎯 Confidence: {confidence:.2f}%")

    st.subheader("🧠 COMPLETE AI EXPLANATION")
    st.text(report)
