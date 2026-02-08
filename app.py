import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="AI Ticket Classifier",
    page_icon="📝",
    layout="centered"
)

# ----------------------------
# Load model & tokenizer (cached)
# ----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert_model/tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("bert_model/model")
    model.to("cpu")          # ensure CPU compatibility
    model.eval()             # inference mode
    return tokenizer, model

tokenizer, model = load_model()

# ----------------------------
# Load label map
# ----------------------------
with open("bert_model/label_map.json", "r") as f:
    label_map = json.load(f)

# convert label ids to int
id2label = {int(v): k for k, v in label_map.items()}

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📝 AI Ticket Classifier")
st.write(
    "This application uses a **Transformer-based NLP model** to automatically "
    "classify customer support tickets."
)

text = st.text_area(
    "Enter your ticket text below:",
    height=150,
    placeholder="Example: My internet connection has been down since yesterday..."
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Category"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some ticket text.")
    else:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        pred_id = torch.argmax(probs, dim=1).item()
        predicted_label = id2label[pred_id]

        # ----------------------------
        # Display result
        # ----------------------------
        st.success(f"✅ **Predicted Category:** {predicted_label}")

        st.subheader("📊 Confidence Scores")
        confidence_scores = {
            id2label[i]: round(float(probs[0][i]) * 100, 2)
            for i in range(len(probs[0]))
        }
        st.json(confidence_scores)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built using Python, PyTorch, Hugging Face & Streamlit")
