# app.py
import streamlit as st
import joblib
import json
import os
import re
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns

# For LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json as _json

# For transformer inference
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

st.set_page_config(page_title="Fake News Detector (Multi-model + XAI)", layout="wide")

# -------------------------
# Helpers
# -------------------------
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_models():
    models = {}
    # Logistic Reg + TFIDF
    if os.path.exists("models/logreg_model.pkl") and os.path.exists("models/tfidf_vectorizer.pkl"):
        models['Logistic Regression'] = {
            "type": "sklearn",
            "model": joblib.load("models/logreg_model.pkl"),
            "tfidf": joblib.load("models/tfidf_vectorizer.pkl")
        }
    # Naive Bayes
    if os.path.exists("models/nb_model.pkl") and os.path.exists("models/tfidf_vectorizer.pkl"):
        models['Multinomial NB'] = {
            "type": "sklearn",
            "model": joblib.load("models/nb_model.pkl"),
            "tfidf": joblib.load("models/tfidf_vectorizer.pkl")
        }
    # LSTM
    if os.path.exists("models/lstm_model.h5") and os.path.exists("models/tokenizer.json"):
        with open("models/tokenizer.json", "r", encoding="utf-8") as f:
            tok_json = f.read()
        tokenizer = tokenizer_from_json(tok_json)
        lstm = load_model("models/lstm_model.h5")
        models['LSTM (Keras)'] = {"type": "lstm", "model": lstm, "tokenizer": tokenizer, "max_len": 200}
    # DistilBERT
    if os.path.exists("models/distilbert_model") and os.path.exists("models/distilbert_tokenizer"):
        try:
            bert_tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert_tokenizer")
            bert_model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_model")
            bert_model.eval()
            models['DistilBERT'] = {"type": "transformer", "model": bert_model, "tokenizer": bert_tokenizer}
        except Exception as e:
            print("Transformer load failed:", e)

    return models

models = load_models()

# Build simple predict_proba wrappers for LIME
def predict_proba_for_model(name, texts):
    entry = models[name]
    t = [clean_text(x) for x in texts]

    if entry['type'] == "sklearn":
        tfidf = entry['tfidf']
        model = entry['model']
        X = tfidf.transform(t)
        return model.predict_proba(X)  # shape (n,2)

    if entry['type'] == "lstm":
        tokenizer = entry['tokenizer']
        seqs = tokenizer.texts_to_sequences(t)
        Xpad = pad_sequences(seqs, maxlen=entry.get("max_len",200), padding='post', truncating='post')
        probs = entry['model'].predict(Xpad).ravel()
        # return shape (n,2): probs for class 0 and 1
        probs_stack = np.vstack([1-probs, probs]).T
        return probs_stack

    if entry['type'] == "transformer":
        tok = entry['tokenizer']
        model = entry['model']
        enc = tok(list(t), truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    raise ValueError("Unknown model type")

# LIME explainer (we'll use class names Fake=0, Real=1)
lime_explainer = LimeTextExplainer(class_names=["Fake", "Real"])

# -------------------------
# Sidebar: model selection + metrics
# -------------------------
st.sidebar.title("Model & Info")
model_names = list(models.keys())
if len(model_names) == 0:
    st.sidebar.warning("No models found in /models. Run train_all.py first.")
    st.stop()

selected_model = st.sidebar.selectbox("Select model", model_names)

# Load metrics per-model file
metrics = {}
metrics_file = "models/metrics_all.json"
if os.path.exists(metrics_file):
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

st.sidebar.markdown("### Model Metrics (selected)")
m = metrics.get(selected_model.lower().replace(" ", "_"), None)
# metrics_all.json uses keys like 'logistic_regression', 'multinomial_nb', 'lstm', 'distilbert'
key_map = {
    "Logistic Regression": "logistic_regression",
    "Multinomial NB": "multinomial_nb",
    "LSTM (Keras)": "lstm",
    "DistilBERT": "distilbert"
}
mapped = None
if metrics and key_map[selected_model] in metrics:
    mapped = metrics[key_map[selected_model]]
    st.sidebar.write(f"**Accuracy:** {mapped.get('accuracy', 0):.3f}")
    st.sidebar.write(f"**Precision:** {mapped.get('precision', 0):.3f}")
    st.sidebar.write(f"**Recall:** {mapped.get('recall', 0):.3f}")
    st.sidebar.write(f"**F1:** {mapped.get('f1_score', mapped.get('f1',0)):.3f}")
    st.sidebar.write(f"**ROC-AUC:** {mapped.get('roc_auc', 0):.3f}")
    if 'confusion_matrix' in mapped:
        cm = np.array(mapped['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(3,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake","Real"], yticklabels=["Fake","Real"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.sidebar.pyplot(fig)
else:
    st.sidebar.info("Metrics not found for selected model.")

# -------------------------
# Main UI
# -------------------------
st.title("📰 Fake News Detector — Multi-Model + Explainability")
st.write("Choose a model from the sidebar, input Title and Content, then Predict. Use Explain to highlight influential words.")

col1, col2 = st.columns([3,1])
with col1:
    title_input = st.text_input("Title")
    content_input = st.text_area("Content", height=250)
with col2:
    st.markdown("### Actions")
    if st.button("Predict"):
        if not (title_input.strip() or content_input.strip()):
            st.warning("Enter title or content.")
        else:
            combined = clean_text((title_input or "") + " " + (content_input or ""))
            try:
                probs = predict_proba_for_model(selected_model, [combined])[0]
                pred = int(np.argmax(probs))
                conf = probs[pred]
                label = "✅ Real News" if pred == 1 else "❌ Fake News"
                if pred == 1:
                    st.success(f"{label} (Confidence: {conf*100:.2f}%)")
                else:
                    st.error(f"{label} (Confidence: {conf*100:.2f}%)")
                # store for explanation button
                st.session_state["last_input"] = combined
                st.session_state["last_pred_probs"] = probs
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if st.button("Explain Prediction"):
        if "last_input" not in st.session_state:
            st.warning("Make a prediction first.")
        else:
            text_to_explain = st.session_state["last_input"]
            try:
                # LIME expects a function that returns proba given list[str]
                predict_fn = lambda texts: predict_proba_for_model(selected_model, texts)
                exp = lime_explainer.explain_instance(text_to_explain, predict_fn, num_features=10)

                # highlight words in text
                def highlight_text(text, exp):
                    highlighted = text
                    for word, weight in exp.as_list():
                        safe = re.escape(word)
                        color = "green" if weight > 0 else "red"
                        highlighted = re.sub(
                            safe,
                            f"<span style='background-color:{color}; color:white; padding:2px;'>{word}</span>",
                            highlighted, flags=re.IGNORECASE)
                    return highlighted

                st.write("### Highlights (green supports Real, red supports Fake)")
                st.markdown(highlight_text(text_to_explain, exp), unsafe_allow_html=True)

                # Also show contribution table and bar chart
                contrib = exp.as_list()
                dfc = pd.DataFrame(contrib, columns=["word","weight"])
                st.write("#### Contributions")
                st.dataframe(dfc)

                fig, ax = plt.subplots()
                dfc['sign'] = dfc['weight'].apply(lambda x: 'pos' if x>0 else 'neg')
                ax.barh(dfc['word'], dfc['weight'])
                ax.set_xlabel("Contribution (positive -> Real)")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Explanation failed: {e}")

# Footer
st.caption("Models trained: Logistic Regression, Multinomial NB, LSTM (Keras), DistilBERT (optional). LIME used for per-prediction explainability.")
