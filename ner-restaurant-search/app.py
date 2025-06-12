import streamlit as st
import pandas as pd
from transformers import pipeline
from typing import List

# --- Config ---
st.set_page_config(page_title="Restaurant NER", layout="centered")

# --- Colors for Entity Types ---
ENTITY_COLORS = {
  "Restaurant_Name": "#4CAF50",   # green
  "Location": "#FF9800",          # orange
  "Dish": "#2196F3",              # blue
  "Rating": "#9C27B0",            # purple
  "Price": "#795548",             # brown
  "Hours": "#607D8B",             # gray-blue
  "Amenity": "#E91E63",           # pink
  "Cuisine": "#009688"            # teal
}

# --- Load TinyBERT NER Model ---
@st.cache_resource
def load_ner_pipeline():
  model_path = "../models/ner_tinybert" 
  classifier = pipeline('token-classification', model=model_path, aggregation_strategy="simple")
  return classifier

ner_pipeline = load_ner_pipeline()

# --- Highlighting Function ---
def highlight_entities(text: str, entities: List[dict]) -> str:
  # Sort entities by start position
  sorted_ents = sorted(entities, key=lambda x: x['start'])
  highlighted_text = ""
  prev_end = 0

  for ent in sorted_ents:
    start, end = ent["start"], ent["end"]
    label = ent["entity_group"].replace("B-", "").replace("I-", "")
    color = ENTITY_COLORS.get(label, "#BDBDBD")  # fallback gray
    word = text[start:end]

    # Add text before entity
    highlighted_text += text[prev_end:start]
    # Add highlighted span
    highlighted_text += f'<span style="background-color:{color}; padding:2px 4px; border-radius:4px; color:white;">{word}</span>'
    prev_end = end

  # Add remaining text
  highlighted_text += text[prev_end:]
  return highlighted_text

# --- Display Entity Legend ---
def show_entity_legend():
  st.markdown("### 🏷️ Entity Color Legend")
  cols = st.columns(4)
  for i, (entity, color) in enumerate(ENTITY_COLORS.items()):
    with cols[i % 4]:
      st.markdown(
        f'<div style="background-color:{color}; color:white; padding:6px 8px; border-radius:6px; text-align:center;">{entity}</div>',
        unsafe_allow_html=True
      )

# --- Display Table with Confidence Scores ---
def show_entity_table(entities: List[dict]):
  if not entities:
    st.info("No entities found.")
    return

  # Create DataFrame
  rows = []
  for ent in entities:
    label = ent["entity_group"].replace("B-", "").replace("I-", "")
    rows.append({
      "Entity": label,
      "Text": ent["word"],
      "Start": ent["start"],
      "End": ent["end"],
      "Score": round(ent["score"], 4)
    })

  df = pd.DataFrame(rows)
  st.markdown("### 📊 Entity Confidence Scores")
  st.dataframe(df, use_container_width=True)
    
# --- UI ---
st.title("🍽️ Restaurant Query NER")
st.markdown("Enter a sentence related to restaurant search and see highlighted named entities like **Dish**, **Location**, and **Rating**.")

show_entity_legend()

st.markdown("---")
text_input = st.text_area("Enter your query:", height=70)

if st.button("Extract Entities"):
  if not text_input.strip():
    st.warning("Please enter some text.")
  else:
    with st.spinner("Extracting entities..."):
      result = ner_pipeline(text_input)

    st.subheader("🔍 Extracted Entities")
    
    st.markdown(highlight_entities(text_input, result), unsafe_allow_html=True)
    st.markdown("---")
    show_entity_table(result)
    