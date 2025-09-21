# 📰 Fake News Detection (Multi-Model + Explainable AI)

This project is a **Fake News Detection Web App** built with **Streamlit**, which integrates **multiple machine learning & deep learning models** along with **Explainable AI (LIME)** for interpretability.  

Users can test news articles (title + content), choose a model, and see both predictions and explanations of which words influenced the decision.  

---

## 🚀 Features

- ✅ **Multiple Models**:
  - Logistic Regression (TF-IDF)
  - Multinomial Naive Bayes (TF-IDF)
  - LSTM (Keras, word embeddings)
  - Transformer (DistilBERT, HuggingFace)

- ✅ **Explainable AI** with **LIME**  
  - Highlights words influencing predictions  
  - Shows contribution weights in tables & charts  

- ✅ **Interactive Streamlit Interface**  
  - Select model from sidebar  
  - View model metrics (Accuracy, Precision, Recall, F1, ROC-AUC)  
  - Visualize confusion matrices  S

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/soloshowdown/fake-news-detection.git
   cd fake-news-detection
2. Install dependencies:
    ```bash
    pip install -r requirements.txt


# requirements.txt

        numpy
        pandas
        scikit-learn
        matplotlib
        seaborn
        nltk
        flask
        lime
        tensorflow==2.15.0
        torch
        transformers==4.41.2
        tf-keras
        joblib



# 🔍 Explainable AI (LIME)

        Green highlights → words supporting "Real News"

        Red highlights → words supporting "Fake News"

        Also includes:

        Contribution weight table

        Bar chart of influential words


# 📈 Results

        Each model is evaluated with:

        Accuracy

        Precision

        Recall

        F1-score

        ROC-AUC

        Confusion Matrix (visualized in sidebar)