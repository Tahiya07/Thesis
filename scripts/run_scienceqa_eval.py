import json
import os
import random

import numpy as np
import pandas as pd

from evaluation.metrics import UnifiedEvaluator
from scripts.run_system import AcademicSystem
from src.privacy.privacy_model import build_default_privacy_model
from src.rag_engine import RAGEngine


MODEL_PATH = "models/qwen.gguf"
VAL_PATH = "models/external_datasets/scienceqa_val.csv"
TEST_PATH = "models/external_datasets/scienceqa_test.csv"
OUTPUT_PATH = "models/scienceqa_eval_results.json"


def load_scienceqa(sample_size=40, seed=42):
    frames = []
    for path in [VAL_PATH, TEST_PATH]:
        if os.path.exists(path):
            frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError("ScienceQA files not found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Context", "Question", "Answer"])
    df["Context"] = df["Context"].astype(str).str.strip()
    df["Question"] = df["Question"].astype(str).str.strip()
    df["Answer"] = df["Answer"].astype(str).str.strip()

    # Keep only cleaner examples with non-trivial context and answer.
    df = df[
        (df["Context"].str.len() > 300) &
        (df["Question"].str.len() > 10) &
        (df["Answer"].str.len() > 2)
    ]

    rnd = random.Random(seed)
    indices = list(df.index)
    rnd.shuffle(indices)
    df = df.loc[indices[:sample_size]].reset_index(drop=True)
    return df


def reset_rag(system):
    system.rag = RAGEngine(privacy_model=build_default_privacy_model())


def main():
    dataset = load_scienceqa()
    system = AcademicSystem(MODEL_PATH)
    evaluator = UnifiedEvaluator(MODEL_PATH)

    results = []
    for _, row in dataset.iterrows():
        reset_rag(system)
        system.rag.add_text(row["Context"])
        res = evaluator.evaluate_sample(
            system,
            row["Question"],
            row["Answer"],
            []
        )
        results.append(res)

    summary = evaluator.summary(results)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": int(len(dataset)),
                "summary": summary,
                "results": results
            },
            f,
            indent=2
        )

    print(json.dumps({"num_samples": len(dataset), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
