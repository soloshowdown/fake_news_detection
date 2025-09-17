# train_all.py
import json
import os
import re
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# For LSTM
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# For transformer
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
)
import torch
from datasets import Dataset

# ---------------------------
# Utility and preprocessing
# ---------------------------
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

os.makedirs("models", exist_ok=True)

print("Loading dataset from HuggingFace...")
ds = load_dataset("Pulk17/Fake-News-Detection-dataset", split="train")
df = pd.DataFrame(ds)
df = df.dropna(subset=["text","label"])
df['text'] = df['text'].map(clean_text)
df['title'] = df['title'].map(clean_text)
df['combined'] = df['title'].fillna("") + " " + df['text'].fillna("")

X = df['combined']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results = {}

# ---------------------------
# Model 1: Logistic Regression
# ---------------------------
print("\nTraining Logistic Regression...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
pred_lr = lr.predict(X_test_tfidf)
proba_lr = lr.predict_proba(X_test_tfidf)[:,1]

metrics_lr = {
    "accuracy": accuracy_score(y_test, pred_lr),
    "precision": precision_score(y_test, pred_lr),
    "recall": recall_score(y_test, pred_lr),
    "f1_score": f1_score(y_test, pred_lr),
    "roc_auc": roc_auc_score(y_test, proba_lr),
    "confusion_matrix": confusion_matrix(y_test, pred_lr).tolist(),
    "classification_report": classification_report(y_test, pred_lr, output_dict=True)
}
results['logistic_regression'] = metrics_lr
joblib.dump(lr, "models/logreg_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
print("LogReg metrics:", metrics_lr)

# ---------------------------
# Model 2: Multinomial Naive Bayes
# ---------------------------
print("\nTraining Multinomial Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
pred_nb = nb.predict(X_test_tfidf)
proba_nb = nb.predict_proba(X_test_tfidf)[:,1]

metrics_nb = {
    "accuracy": accuracy_score(y_test, pred_nb),
    "precision": precision_score(y_test, pred_nb),
    "recall": recall_score(y_test, pred_nb),
    "f1_score": f1_score(y_test, pred_nb),
    "roc_auc": roc_auc_score(y_test, proba_nb),
    "confusion_matrix": confusion_matrix(y_test, pred_nb).tolist(),
    "classification_report": classification_report(y_test, pred_nb, output_dict=True)
}
results['multinomial_nb'] = metrics_nb
joblib.dump(nb, "models/nb_model.pkl")
print("NB metrics:", metrics_nb)

# ---------------------------
# Model 3: LSTM (Keras)
# ---------------------------
print("\nTraining LSTM (Keras)...")
# Use smaller vocab and sequence length to keep training quick
MAX_VOCAB = 20000
MAX_LEN = 200
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Simple LSTM model
lstm_model = Sequential([
    Embedding(MAX_VOCAB, EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train briefly (adjust epochs for your machine)
lstm_model.fit(X_train_pad, y_train, epochs=3, batch_size=64, validation_split=0.1)

pred_lstm_prob = lstm_model.predict(X_test_pad).ravel()
pred_lstm = (pred_lstm_prob >= 0.5).astype(int)

metrics_lstm = {
    "accuracy": accuracy_score(y_test, pred_lstm),
    "precision": precision_score(y_test, pred_lstm),
    "recall": recall_score(y_test, pred_lstm),
    "f1_score": f1_score(y_test, pred_lstm),
    "roc_auc": roc_auc_score(y_test, pred_lstm_prob),
    "confusion_matrix": confusion_matrix(y_test, pred_lstm).tolist(),
    "classification_report": classification_report(y_test, pred_lstm, output_dict=True)
}
results['lstm'] = metrics_lstm

# save tokenizer + model
tokenizer_json = tokenizer.to_json()
with open("models/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_json)
lstm_model.save("models/lstm_model.h5")
print("LSTM metrics:", metrics_lstm)

# ---------------------------
# Model 4: DistilBERT (fine-tune)  -- OPTIONAL heavy
# ---------------------------
do_transformer = True  # set False to skip
if do_transformer:
    try:
        print("\nPreparing DistilBERT fine-tuning (this can be slow)...")
        transformer_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        def tokenize_batch(texts):
            return transformer_tokenizer(texts, truncation=True, padding='max_length', max_length=128)

        ds_train = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
        ds_test = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

        ds_train = ds_train.map(lambda x: tokenize_batch(x["text"]), batched=True)
        ds_train = ds_train.remove_columns(["text"])
        ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        ds_test = ds_test.map(lambda x: tokenize_batch(x["text"]), batched=True)
        ds_test = ds_test.remove_columns(["text"])
        ds_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        model_transformer = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

        training_args = TrainingArguments(
            output_dir="models/distilbert_out",
            num_train_epochs=1,     # use 1-2 epochs for speed; increase if you have time/GPU
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy"
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            probs = torch.softmax(torch.tensor(logits), dim=1)[:,1].numpy()
            return {
                "accuracy": (preds == labels).astype(float).mean(),
                "precision": float( ( (preds == 1) & (labels == 1) ).sum() / max(1, (preds == 1).sum()) ),
                "recall": float( ( (preds == 1) & (labels == 1) ).sum() / max(1, (labels == 1).sum()) ),
                "f1": 0.0  # HF trainer doesn't need exact f1 here; we'll compute below
            }

        trainer = Trainer(
            model=model_transformer,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_test,
            compute_metrics=compute_metrics
        )

        trainer.train()

        # Evaluate predictions on test
        preds_output = trainer.predict(ds_test)
        logits = preds_output.predictions
        preds = np.argmax(logits, axis=-1)
        probs = torch.softmax(torch.tensor(logits), dim=1)[:,1].numpy()

        metrics_tf = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "classification_report": classification_report(y_test, preds, output_dict=True)
        }
        results['distilbert'] = metrics_tf

        # Save transformer model + tokenizer
        model_transformer.save_pretrained("models/distilbert_model")
        transformer_tokenizer.save_pretrained("models/distilbert_tokenizer")
        print("DistilBERT metrics:", metrics_tf)
    except Exception as e:
        print("Transformer training failed or skipped:", e)
        results['distilbert'] = {"error": str(e)}

# ---------------------------
# Save consolidated metrics
# ---------------------------
with open("models/metrics_all.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nAll training done. Models and metrics saved into /models folder.")
