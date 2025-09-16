from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import joblib

# 1. Load dataset
dataset = load_dataset("Pulk17/Fake-News-Detection-dataset", split="train")
df = pd.DataFrame(dataset)

# 2. Clean text
def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df['text'] = df['text'].fillna("").map(clean_text)
df['title'] = df['title'].fillna("").map(clean_text)
df['combined'] = df['title'] + " " + df['text']

# 3. Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 4. TF-IDF features
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = tfidf.fit_transform(train_df['combined'])
X_test = tfidf.transform(test_df['combined'])
y_train, y_test = train_df['label'], test_df['label']

# 5. Train Logistic Regression
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# 6. Evaluate
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# 7. Save model + vectorizer
joblib.dump(model, "logreg_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
