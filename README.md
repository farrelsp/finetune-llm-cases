# 🧠 Fine-tune LLM Cases

This repository contains a collection of **use-case-specific fine-tuning experiments** on various datasets using **Large Language Models (LLMs)**. Each folder represents a distinct project, complete with datasets, Jupyter notebooks, and in some cases, **Streamlit apps** to demonstrate the results.

## 💡 Key Highlights

- Each folder includes:
  - 🧪 **Jupyter notebooks** for training, evaluation, and experimentation
  - 📊 Dataset files (in `datasets/` folder)
  - 🌐 **Streamlit apps** for interactive exploration 

- Focused on **practical applications** of LLMs and transfer learning in NLP and Computer Vision.

## 🚀 Getting Started

1. **Clone the repository**
    ```
    git clone https://github.com/farrelsp/finetune-llm-cases.git
    cd finetune-llm-cases
    ```
2. **Set up your environment**
    - Use `conda` or `virtualenv`
    - Recommended: Python 3.10+
    - Install required packages per folder (usually in requirements.txt or in the notebook headers)

3. **Run any case**
    - Navigate into the desired folder and launch the notebook or Streamlit app
    
        ```
        cd tweets-sentiment
        streamlit run app.py  # if available
        ```

## 🧩 Use Cases (Details)
1. **📰 Fake News Detection**

    Benchmarking three different BERT models to detect fake news.

2. **😃 Tweets Sentiment Analysis**

    Training a sentiment classifier for tweets using BERT.

3. **🍛 Indian Food Classification**

    Image classification task to identify Indian dishes using Vision Transformer model.

4. **🛍️ LoRA Product Generation**

    Generating product name and descriptions using a fine-tuned Microsoft Phi-2 model with Low-Rank Adaptation (LoRA).

5. **🍽️ NER for Restaurant Search**

    Custom NER training to extract restaurant names, dishes, and locations from user input.

6. **📝 Text Summarization**

   Use custom dataset to summarize dialogues with T5 model.
    
7. **💬 QLoRA Chat Model**

    Experimenting with quantized LoRA-based fine-tuning to build a TinyLlama as a chat model.
