import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_PATH = "data/obe_dataset.csv"
REPORT_PATH = "models/bloom_benchmark_report.json"
LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def build_dataset():
    df = pd.read_csv(DATA_PATH).dropna(subset=["question", "bloom_level"])
    df["bloom_level"] = df["bloom_level"].astype(str).str.strip().str.title()
    df = df[df["bloom_level"].isin(LEVELS)]

    return {
        "question_only": df["question"].astype(str),
        "question_plus_meta": (
            "Subject: " + df["subject"].astype(str) +
            " Topic: " + df["topic"].astype(str) +
            " Subtopic: " + df["subtopic"].astype(str) +
            " Difficulty: " + df["difficulty"].astype(str) +
            " Language: " + df["language"].astype(str) +
            " CognitiveSkill: " + df["cognitive_skill"].astype(str) +
            " SourceType: " + df["source_type"].astype(str) +
            " Question: " + df["question"].astype(str)
        ),
        "full_prompt": (
            "Subject: " + df["subject"].astype(str) +
            " Topic: " + df["topic"].astype(str) +
            " Subtopic: " + df["subtopic"].astype(str) +
            " Difficulty: " + df["difficulty"].astype(str) +
            " Language: " + df["language"].astype(str) +
            " CognitiveSkill: " + df["cognitive_skill"].astype(str) +
            " SourceType: " + df["source_type"].astype(str) +
            " Summary: " + df["summary"].astype(str) +
            " Question: " + df["question"].astype(str) +
            " Answer: " + df["answer"].astype(str)
        ),
        "labels": df["bloom_level"],
    }


def main():
    data = build_dataset()
    y = data["labels"]

    models = {
        "logreg_word": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=300, solver="saga", class_weight="balanced"))
        ]),
        "sgd_word": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=70000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-5, penalty="l2", max_iter=30, tol=1e-3, class_weight="balanced"))
        ]),
        "svm_charword": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=120000, ngram_range=(1, 2), analyzer="char_wb", min_df=2, sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced"))
        ]),
    }

    report = {}

    for feature_name in ["question_only", "question_plus_meta", "full_prompt"]:
        X = data[feature_name]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )

        report[feature_name] = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            report[feature_name][model_name] = {
                "accuracy": float(accuracy_score(y_val, pred))
            }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
