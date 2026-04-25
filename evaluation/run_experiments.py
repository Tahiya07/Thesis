import json
import os
import datetime
import numpy as np

from evaluation.metrics import UnifiedEvaluator
from scripts.run_system import AcademicSystem


MODEL_PATH = "models/qwen.gguf"

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"models/experiment_runs/run_{run_id}"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


DATASET = [
    {
        "question": "What is the main purpose of the fire-fighting and gas detection robot?",
        "reference": "To detect fire and gas leakage and assist in emergency response",
        "keywords": ["fire", "gas", "detection", "robot"]
    },
    {
        "question": "What problem does the robot aim to solve?",
        "reference": "Fire accidents and gas leakage hazards that threaten safety",
        "keywords": ["fire", "gas leak", "hazard", "safety"]
    },
    {
        "question": "What are the main components of the robot system?",
        "reference": "ESP32 microcontroller, MQ-2 gas sensor, flame sensors, motors, and water pump",
        "keywords": ["sensor", "ESP32", "motor", "pump"]
    },
    {
        "question": "How does the robot detect fire or gas?",
        "reference": "Using flame sensors for fire and MQ-2 gas sensor for gas detection",
        "keywords": ["flame sensor", "MQ-2", "gas"]
    },
    {
        "question": "What happens when gas is detected?",
        "reference": "The system triggers an alert such as a buzzer or notification",
        "keywords": ["alert", "buzzer", "alarm"]
    }
]


def run():
    print("\nStarting Publication-Grade Evaluation Pipeline")
    print(f"Output: {OUTPUT_DIR}")

    system = AcademicSystem(MODEL_PATH)
    evaluator = UnifiedEvaluator(MODEL_PATH)

    pdf_path = "data/robot_proposal.pdf"
    if os.path.exists(pdf_path):
        system.add_pdf(pdf_path)

    image_path = "data/ckt1.png"
    if os.path.exists(image_path):
        system.add_image(image_path)

    results = []
    for i, sample in enumerate(DATASET):
        print(f"\nQ{i + 1}: {sample['question']}")
        res = evaluator.evaluate_sample(
            system,
            sample["question"],
            sample.get("reference"),
            sample.get("keywords")
        )
        results.append(res)

    attack = evaluator.attack_simulation(system=system, n=10)
    attack_rate = attack["attack_success_rate"]

    lambdas = [0.0, 0.2, 0.5]
    quality_curve = []
    privacy_curve = []

    for lam in lambdas:
        system.rag.set_ablation(lambda_privacy=lam)
        lam_results = []
        for sample in DATASET:
            lam_results.append(
                evaluator.evaluate_sample(
                    system,
                    sample["question"],
                    sample.get("reference"),
                    sample.get("keywords")
                )
            )

        quality_curve.append(float(np.mean([r["answer_similarity"] for r in lam_results])))
        privacy_curve.append(float(np.mean([r["mean_chunk_privacy"] for r in lam_results])))

    ablations = {}
    ablation_settings = {
        "full_system": {
            "use_privacy": True,
            "use_diversity": True,
            "use_rejection": True,
            "use_learning_privacy": system.rag.use_learning_privacy,
            "lambda_privacy": system.rag.lambda_privacy,
        },
        "no_privacy": {
            "use_privacy": False,
            "use_diversity": True,
            "use_rejection": True,
            "use_learning_privacy": system.rag.use_learning_privacy,
            "lambda_privacy": system.rag.lambda_privacy,
        },
        "no_diversity": {
            "use_privacy": True,
            "use_diversity": False,
            "use_rejection": True,
            "use_learning_privacy": system.rag.use_learning_privacy,
            "lambda_privacy": system.rag.lambda_privacy,
        },
        "no_rejection": {
            "use_privacy": True,
            "use_diversity": True,
            "use_rejection": False,
            "use_learning_privacy": system.rag.use_learning_privacy,
            "lambda_privacy": system.rag.lambda_privacy,
        },
    }

    for name, cfg in ablation_settings.items():
        system.rag.set_ablation(**cfg)
        variant_results = []
        for sample in DATASET:
            variant_results.append(
                evaluator.evaluate_sample(
                    system,
                    sample["question"],
                    sample.get("reference"),
                    sample.get("keywords")
                )
            )

        ablations[name] = {
            "config": cfg,
            "summary": evaluator.summary(variant_results)
        }

    system.rag.set_ablation(
        use_privacy=True,
        use_diversity=True,
        use_rejection=True,
        lambda_privacy=0.3
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "attack_success_rate": attack_rate,
            "ablations": ablations,
            "privacy_accuracy_curve": {
                "lambdas": lambdas,
                "answer_similarity": quality_curve,
                "mean_chunk_privacy": privacy_curve
            }
        }, f, indent=2)

    print("\nSUMMARY")
    print("Latency:", np.mean([r["latency"] for r in results]))
    print("Answer Similarity:", np.mean([r["answer_similarity"] for r in results]))
    print("Faithfulness:", np.mean([r["faithfulness"] for r in results]))
    print("Hallucination:", np.mean([r["hallucination"] for r in results]))
    print("Precision@k:", np.mean([r["precision@k"] for r in results]))
    print("Retrieval Redundancy:", np.mean([r["retrieval_redundancy"] for r in results]))
    print("Mean Chunk Privacy:", np.mean([r["mean_chunk_privacy"] for r in results]))
    print("Privacy Leakage:", np.mean([r["privacy_leakage"] for r in results]))
    print("Rejection Rate:", np.mean([r["rejection"] for r in results]))
    print("Bloom Confidence:", np.mean([r["bloom_confidence"] for r in results]))
    print("Bloom Uncertainty:", np.mean([r["bloom_uncertainty"] for r in results]))
    print("\nAttack Success Rate:", attack_rate)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
