import glob
import json
import os

import matplotlib.pyplot as plt


OUTPUT_DIR = "models/final_plots"


def find_latest_results():
    candidates = sorted(glob.glob("models/experiment_runs/run_*/results.json"))
    if not candidates:
        raise FileNotFoundError("No experiment results found in models/experiment_runs")
    return candidates[-1]


def load_results(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_ablation_metrics(data):
    ablations = data["ablations"]
    settings = list(ablations.keys())
    answer_similarity = [ablations[s]["summary"]["answer_similarity"] for s in settings]
    mean_privacy = [ablations[s]["summary"]["mean_chunk_privacy"] for s in settings]
    redundancy = [ablations[s]["summary"]["retrieval_redundancy"] for s in settings]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(settings, answer_similarity, color="#4C78A8")
    axes[0].set_title("Answer Similarity by Ablation")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(settings, mean_privacy, color="#F58518")
    axes[1].set_title("Mean Chunk Privacy by Ablation")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(settings, redundancy, color="#54A24B")
    axes[2].set_title("Retrieval Redundancy by Ablation")
    axes[2].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "ablation_summary.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_privacy_quality_curve(data):
    curve = data["privacy_accuracy_curve"]
    lambdas = [point["lambda"] for point in curve]
    answer_similarity = [point["answer_similarity"] for point in curve]
    mean_privacy = [point["mean_chunk_privacy"] for point in curve]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(lambdas, answer_similarity, marker="o", color="#4C78A8", label="Answer Similarity")
    ax1.set_xlabel("Lambda Privacy")
    ax1.set_ylabel("Answer Similarity", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")

    ax2 = ax1.twinx()
    ax2.plot(lambdas, mean_privacy, marker="s", color="#E45756", label="Mean Chunk Privacy")
    ax2.set_ylabel("Mean Chunk Privacy", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")

    plt.title("Privacy-Quality Trade-off")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "privacy_quality_curve.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_main_metrics(data):
    summary = data.get("summary") or {}
    metrics = {
        "Answer Similarity": summary.get("answer_similarity", 0.0),
        "Faithfulness": summary.get("faithfulness", 0.0),
        "Precision@k": summary.get("precision@k", 0.0),
        "Bloom Accuracy": summary.get("bloom_accuracy", 0.0),
        "Privacy Leakage": summary.get("privacy_leakage", 0.0),
        "Rejection Rate": summary.get("rejection", 0.0),
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        list(metrics.keys()),
        list(metrics.values()),
        color=["#4C78A8", "#72B7B2", "#54A24B", "#B279A2", "#E45756", "#9D755D"]
    )
    ax.set_title("Main Evaluation Metrics")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "main_metrics.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = find_latest_results()
    data = load_results(results_path)
    outputs = {
        "results_path": results_path,
        "ablation_summary": plot_ablation_metrics(data),
        "privacy_quality_curve": plot_privacy_quality_curve(data),
        "main_metrics": plot_main_metrics(data),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
