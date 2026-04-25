import os
import pickle
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


MODEL_PATH = "models/bloom_ldl.pkl"
REPORT_PATH = "models/bloom_training_report.json"

LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

BT_MAP = {
    "Knowledge": "Remember",
    "Comprehension": "Understand",
    "Application": "Apply",
    "Analysis": "Analyze",
    "Synthesis": "Create",
    "Evaluation": "Evaluate",
    "Remember": "Remember",
    "Understand": "Understand",
    "Apply": "Apply",
    "Analyze": "Analyze",
    "Evaluate": "Evaluate",
    "Create": "Create",
}


def load_exam_bloom_dataset():
    path = "models/external_datasets/exam_combined_dataset.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["question"] = df["QUESTION"].astype(str).str.strip()
    df["bloom_level"] = df["BT LEVEL"].astype(str).str.strip().str.title().map(BT_MAP)
    df = df.dropna(subset=["question", "bloom_level"])
    df = df[df["bloom_level"].isin(LEVELS)]
    return df[["question", "bloom_level"]], path


def load_obe_dataset():
    path = "data/obe_dataset.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["question", "bloom_level"])
    df["question"] = df["question"].astype(str).str.strip()
    df["bloom_level"] = df["bloom_level"].astype(str).str.strip().str.title()
    df = df[df["bloom_level"].isin(LEVELS)]
    return df[["question", "bloom_level"]], path


def load_data():
    external_path = "models/external_datasets/exam_combined_dataset.csv"
    if os.path.exists(external_path):
        return load_exam_bloom_dataset()
    return load_obe_dataset()


def train():
    df, source_path = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        df["question"],
        df["bloom_level"],
        test_size=0.2,
        random_state=42,
        stratify=df["bloom_level"]
    )

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(
        max_iter=400,
        solver="saga",
        class_weight="balanced",
        verbose=0
    )
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_val_vec)
    acc = accuracy_score(y_val, preds)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "model": model,
                "levels": list(model.classes_),
                "validation_accuracy": float(acc),
                "source_dataset": source_path
            },
            f
        )

    report = classification_report(y_val, preds, digits=4, output_dict=True)
    report["validation_accuracy"] = float(acc)
    report["num_train"] = int(len(X_train))
    report["num_val"] = int(len(X_val))
    report["source_dataset"] = source_path

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved report to {REPORT_PATH}")
    print(f"Source dataset: {source_path}")
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_val, preds, digits=4))


if __name__ == "__main__":
    train()
