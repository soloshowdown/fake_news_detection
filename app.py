import streamlit as st
import joblib
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Create pipeline for LIME
pipeline = make_pipeline(tfidf, model)

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=["Fake", "Real"])

# Function to highlight words
def highlight_text(text, exp):
    highlighted = text
    for word, weight in exp.as_list():
        safe_word = re.escape(word)
        color = "green" if weight > 0 else "red"
        highlighted = re.sub(
            safe_word,
            f"<span style='background-color:{color}; color:white; padding:2px;'>{word}</span>",
            highlighted,
            flags=re.IGNORECASE
        )
    return highlighted

# Streamlit App
st.title("📰 Fake News Detection with Explainable AI")

title = st.text_input("Enter News Title")
content = st.text_area("Enter News Content")

if st.button("Predict"):
    if title or content:
        combined_input = title + " " + content
        prediction = model.predict(tfidf.transform([combined_input]))[0]
        proba = model.predict_proba(tfidf.transform([combined_input]))[0]

        label = "🟥 Fake News" if prediction == 0 else "🟩 Real News"
        st.write(f"### Prediction: {label}")
        st.write(f"Confidence: {max(proba)*100:.2f}%")
        
        st.session_state['last_input'] = combined_input
    else:
        st.warning("Please enter some text.")

# Explain Prediction
if 'last_input' in st.session_state and st.button("Explain Prediction"):
    text = st.session_state['last_input']
    exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=10)

    st.write("### Explanation (word highlights):")
    highlighted_text = highlight_text(text, exp)
    st.markdown(highlighted_text, unsafe_allow_html=True)

# Sidebar: Metrics
st.sidebar.title("📊 Model Metrics")

try:
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    st.sidebar.write(f"**Accuracy:** {metrics['accuracy']:.2f}")
    st.sidebar.write(f"**Precision:** {metrics['precision']:.2f}")
    st.sidebar.write(f"**Recall:** {metrics['recall']:.2f}")
    st.sidebar.write(f"**F1 Score:** {metrics['f1_score']:.2f}")
    st.sidebar.write(f"**ROC-AUC:** {metrics['roc_auc']:.2f}")

    # Confusion Matrix Heatmap
    st.sidebar.write("### Confusion Matrix")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.sidebar.pyplot(fig)

except:
    st.sidebar.warning("Metrics not found. Train the model first.")
