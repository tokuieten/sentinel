# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import numpy as np
import html

# === Config ===
MODEL_DIR = "tokuieten/sentinel-distilbert"  # change if your model is in a different folder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["negative", "positive"]
MAX_LENGTH = 256

st.set_page_config(page_title="DeepSentiment — Demo", layout="centered")


# Cache the heavy resource (model + tokenizer)
@st.cache_resource
def load_model_and_tokenizer():
    # Try local model dir first; if missing, HF will attempt to download from hub (requires internet)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=False)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def predict_proba(texts):
    """LIME expects a function texts -> numpy array of shape (n_samples, n_classes)."""
    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    return probs

# Small helper to produce highlighted HTML using LIME weights
def lime_highlight_html(text, exp):
    # exp.as_list() returns tokens (words) mapped to weights for the predicted class (positive weight -> positive)
    weights = dict(exp.as_list())
    words = text.split()
    parts = []
    for w in words:
        key = w.lower().strip(".,!?;:\"'()[]{}")
        weight = weights.get(key, 0.0)
        if weight > 0.0:
            # positive contribution: light green background
            # scale alpha by weight (clamped)
            alpha = min(0.85, 0.2 + abs(weight) * 0.25)
            style = f"background: rgba(144,238,144,{alpha}); padding:2px; border-radius:3px; margin:1px"
        elif weight < 0.0:
            # negative contribution: light red
            alpha = min(0.85, 0.2 + abs(weight) * 0.25)
            style = f"background: rgba(255,182,193,{alpha}); padding:2px; border-radius:3px; margin:1px"
        else:
            style = "padding:2px; margin:1px"
        safe = html.escape(w)
        parts.append(f"<span style='{style}'>{safe}</span>")
    return " ".join(parts)

st.title("DeepSentiment — Sentiment Classifier (DistilBERT)")
st.write("Type a sentence or review and press **Predict**. The model returns `positive` or `negative` and highlights contributing words.")

input_text = st.text_area("Enter text here:", value="That movie was amazing and fun to watch!")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Predict"):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            probs = predict_proba([input_text])[0]
            pred_idx = int(np.argmax(probs))
            pred_label = LABELS[pred_idx]

            st.markdown("### Prediction")
            st.write(f"**Predicted label:** {pred_label}")

            # Optionally show confidence if you want (commented)
            # st.write(f"Confidence: {probs[pred_idx]:.3f}")

            # LIME explanation (word contributions)
            st.markdown("### Explanation (LIME) — words that pushed the prediction")
            explainer = LimeTextExplainer(class_names=LABELS)
            # Using num_samples=500 for reasonably fast results; lower to speed up
            exp = explainer.explain_instance(input_text, predict_proba, num_features=10, num_samples=500)

            # Show highlighted text
            html_out = lime_highlight_html(input_text, exp)
            st.write(html_out, unsafe_allow_html=True)

            # Show top features as a small table
            st.markdown("**Top contributing tokens (word, weight)**")
            fm = exp.as_list()
            st.table(fm[:10])

with col2:
    st.markdown("## Tips")
    st.write("- The model predicts **only** positive or negative.")
    st.write("- Use short to medium sentences for clearer explanations.")
    st.write("- If the model is very overconfident, consider calibration (temperature scaling).")

st.markdown("---")
st.markdown("Built with DistilBERT (Hugging Face Transformers) and LIME for explanation.")

