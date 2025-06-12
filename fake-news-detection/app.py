import streamlit as st
from transformers import pipeline

# Set page config
st.set_page_config(page_title="Fake News Classifier", layout="centered")

# Model paths
MODEL_PATHS = {
  "DistilBERT": "../models/fake_news_DistilBERT",
  "MobileBERT": "../models/fake_news_MobileBERT",
  "TinyBERT": "../models/fake_news_TinyBERT"
}

# Labels
COLORS = {
  "Real": "#4CAF50",    # green
  "Fake": "#FF4B4B",   # red
}

# Load model & tokenizer
@st.cache_resource
def load_model(model_path):
  classifier = pipeline('text-classification', model=model_path)
  return classifier

# Prediction function
def predict(model, text):
  outputs = model([text])
  return outputs[0]['label'], outputs[0]['score']

# --- UI Starts Here ---
st.title("📰 Fake News Title Classifier")
st.markdown("Classify news headlines as **Fake** or **Real** using different BERT-based models.")

# Input text
title_input = st.text_area("Enter news title:", height=100)

# Model selection
selected_models = st.multiselect(
  "Choose model(s) to classify with:",
  options=list(MODEL_PATHS.keys()),
  default=["DistilBERT"]
)

# Run prediction
if st.button("Classify"):
  if not title_input.strip():
    st.warning("Please enter a news title.")
  elif not selected_models:
    st.warning("Please select at least one model.")
  else:
    st.subheader("🧠 Model Predictions")
    
    # Create columns dynamically
    cols = st.columns(len(selected_models))
    for i, model_name in enumerate(selected_models):
      model = load_model(MODEL_PATHS[model_name])
      label, scores = predict(model, title_input)
      color = COLORS[label]

      with cols[i]:
        st.markdown(f"### {model_name}")
        st.markdown(
          f"""
          <div style="padding: 1em; border-radius: 10px; background-color: {color}; color: white; text-align: center; font-size: 24px;">
              {label}
          </div>
          """,
          unsafe_allow_html=True
        )
        st.caption(f"Confidence → {scores:.4f}")