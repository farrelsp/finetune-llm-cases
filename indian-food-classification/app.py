import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import pipeline, AutoImageProcessor
import torch

# --- Setup ---
st.set_page_config(page_title="Indian Food Classifier", layout="centered")
st.title("🍛 Indian Food Classifier")
st.markdown("Upload an image or paste an image URL to classify it into **20 Indian food categories** using a Vision Transformer model.")

# --- Load Inference Pipeline ---
@st.cache_resource
def load_pipeline():
  device = 0 if torch.cuda.is_available() else -1
  return pipeline("image-classification", model="../models/food_vit", device=device)

pipe = load_pipeline()

# --- Load Image from URL ---
def load_image_from_url(url):
  try:
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image
  except Exception as e:
    st.error(f"Could not load image: {e}")
    return None

# --- Input Method ---
input_method = st.radio("Choose image input method:", ["Upload Image", "Image URL"])
img = None

if input_method == "Upload Image":
  file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
  if file:
    img = Image.open(file).convert("RGB")

else:
  url = st.text_input("Enter image URL")
  if url:
    img = load_image_from_url(url)

# --- Classification Logic ---
if img:
  st.image(img, caption="Input Image")

  if st.button("🔍 Classify Image"):
    with st.spinner("Classifying..."):
      results = pipe(img)

    top_pred = results[0]
    label = top_pred["label"].replace("_", " ").title()
    st.success(f"🍽️ **Top Prediction:** {label} ({top_pred['score']:.2%})")

    st.markdown("### 🔝 Top 5 Predictions")
    for r in results[:5]:
      label = r["label"].replace("_", " ").title()
      st.write(f"- **{label}**: {r['score']*100:.2f}%")