import datetime
import json
import os

import numpy as np

from evaluation.metrics import UnifiedEvaluator
from scripts.run_system import AcademicSystem


MODEL_PATH = "models/qwen.gguf"
DATASET_PATH = "evaluation/robot_pdf_eval_dataset.json"
PDF_PATH = "data/robot_proposal.pdf"


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_variant(system, evaluator, dataset):
    results = []
    for sample in dataset:
        results.append(
            evaluator.evaluate_sample(
                system,
                sample["question"],
                sample.get("reference"),
                sample.get("keywords"),
                sample.get("expected_bloom"),
            )
        )
    return results


def run():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/experiment_runs/run_{run_id}"
    output_file = os.path.join(output_dir, "results.json")
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(DATASET_PATH)

    print("\nStarting Robot PDF Evaluation Pipeline")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Samples: {len(dataset)}")
    print(f"Output: {output_file}")

    system = AcademicSystem(MODEL_PATH)
    evaluator = UnifiedEvaluator(MODEL_PATH)

    if os.path.exists(PDF_PATH):
        system.add_pdf(PDF_PATH)
    else:
        raise FileNotFoundError(f"Missing PDF: {PDF_PATH}")

    results = evaluate_variant(system, evaluator, dataset)
    summary = evaluator.summary(results)

    attack = evaluator.attack_simulation(system=system, n=10)
    attack_rate = attack["attack_success_rate"]

    lambdas = [0.0, 0.2, 0.5]
    privacy_accuracy_curve = evaluator.privacy_accuracy_curve(
        dataset,
        lambdas,
        system=system
    )

    ablations = {}
    ablation_settings = {
        "full_system": {
            "use_privacy": True,
            "use_diversity": True,
            "use_rejection": True,
            "use_learning_privacy": system.rag.use_learning_privacy,
            "lambda_privacy": 0.3,
        },
        "no_privacy": {
            "use_privacy": False,
            "use_diversity": True,
            "use_rejection": True,
            "use_learning_privacy": system.rag.use_learning_privacy,
            "lambda_privacy": 0.3,
        },
        "no_diversity": {
            "use_privacy": True,
            "use_diversity": False,
            "use_rejection": True,
            "use_learning_privacy": system.rag.use_learning_privacy,
            "lambda_privacy": 0.3,
        },
        "no_rejection": {
            "use_privacy": True,
            "use_diversity": True,
            "use_rejection": False,
            "use_learning_privacy": system.rag.use_learning_privacy,
            "lambda_privacy": 0.3,
        },
    }

    for name, cfg in ablation_settings.items():
        system.rag.set_ablation(**cfg)
        variant_results = evaluate_variant(system, evaluator, dataset)
        ablations[name] = {
            "config": cfg,
            "summary": evaluator.summary(variant_results),
        }

    system.rag.set_ablation(
        use_privacy=True,
        use_diversity=True,
        use_rejection=True,
        use_learning_privacy=system.rag.use_learning_privacy,
        lambda_privacy=0.3,
    )

    payload = {
        "dataset_name": "robot_pdf_eval_dataset",
        "dataset_path": DATASET_PATH,
        "pdf_path": PDF_PATH,
        "num_samples": len(dataset),
        "summary": summary,
        "results": results,
        "attack_success_rate": attack_rate,
        "privacy_accuracy_curve": privacy_accuracy_curve,
        "ablations": ablations,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"attack_success_rate: {attack_rate}")
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    run()
