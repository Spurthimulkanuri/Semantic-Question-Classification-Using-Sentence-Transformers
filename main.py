import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

# Load dataset
def load_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if ",,," in line:
                question, label = line.strip().rsplit(",,,", 1)
                data.append((question.strip(), label.strip()))
    return pd.DataFrame(data, columns=["question", "label"])

# Preprocess and split
def preprocess_and_split(df):
    X = df["question"]
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train, embedder):
    X_train_vec = embedder.encode(X_train.tolist(), convert_to_numpy=True)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return model

# Evaluate
def evaluate_model(model, embedder, X_test, y_test):
    X_test_vec = embedder.encode(X_test.tolist(), convert_to_numpy=True)
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

# Save
def save_model(model, embedder):
    joblib.dump(model, "model.pkl")
    joblib.dump(embedder, "embedder.pkl")  # Not needed as it's a transformer; saving path for consistency

if __name__ == "__main__":
    df = load_data("LabelledData.txt")
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    model = train_model(X_train, y_train, embedder)
    evaluate_model(model, embedder, X_test, y_test)
    save_model(model, embedder)
    print("Model and embedder saved.")

