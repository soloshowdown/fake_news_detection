import streamlit as st
import joblib
import re

# Load model + vectorizer
model = joblib.load("logreg_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning
def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------- Interface ---------------- #
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Check whether a news article is **Real or Fake** using an NLP model (Logistic Regression).")

# Input fields
st.subheader("Enter News Details")
title_input = st.text_input("News Title", placeholder="e.g. Breaking: New vaccine approved for use")
text_input = st.text_area("News Content", placeholder="Paste the full article text here...")

# Button
if st.button("🔍 Check News"):
    if title_input.strip() == "" and text_input.strip() == "":
        st.warning("⚠️ Please enter a title or content.")
    else:
        # Combine title + text
        combined_input = clean_text(title_input + " " + text_input)
        features = tfidf.transform([combined_input])

        # Predict
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        # Display result
        st.markdown("---")
        if prediction == 1:
            st.success(f"✅ **Real News** (Confidence: {proba[1]*100:.2f}%)")
        else:
            st.error(f"❌ **Fake News** (Confidence: {proba[0]*100:.2f}%)")
        
        st.markdown("---")
        st.caption("Model: Logistic Regression with TF-IDF features")
