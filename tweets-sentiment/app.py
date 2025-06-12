import streamlit as st
from transformers import pipeline

st.set_page_config(
  page_title="Tweets Sentiment Classifier",  # <- This is the browser tab title
)

# Load model and tokenizer 
@st.cache_resource
def load_model():
  model_path = "./models/bert-base-sentiment-model"
  classifier = pipeline('text-classification', model=model_path)
  return classifier

classifier = load_model()

# STREAMLIT UI
# Title
st.title("BERT for Tweets Multi-Class Sentiment Classification")
st.markdown("Predict sentiment of a tweet using a fine-tuned BERT model.")

# Input
tweet = st.text_area("Enter tweet text here", height=150)

# Predict
if st.button("Classify"):
  if tweet.strip() == "":
    st.warning("Please enter some text.")
  else:
    outputs = classifier([tweet])
    st.success(f"**Sentiment:** {outputs[0]['label'].capitalize()} (Confidence: {outputs[0]['score']:.4f})")