import json
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================
# CONFIG
# =============================
OUTPUT_DIR = "outputs"
RESULT_FILE = None


def find_latest_results():
    global RESULT_FILE

    runs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("run_")]
    runs.sort(reverse=True)

    latest = runs[0]
    RESULT_FILE = f"{OUTPUT_DIR}/{latest}/results.json"

    return f"{OUTPUT_DIR}/{latest}"


def load_results():
    with open(RESULT_FILE, "r") as f:
        return json.load(f)


def aggregate(results):

    metrics = {
        "latency": [],
        "bleu": [],
        "faithfulness": [],
        "hallucination": [],
        "privacy_leakage": [],
        "uncertainty": [],
        "precision@k": [],
        "bloom_confidence": []
    }

    for r in results:

        # standard metrics
        for k in metrics:
            if k != "bloom_confidence":
                metrics[k].append(r.get(k, 0))

        # BLOOM confidence (safe extraction)
        dist = r.get("bloom_distribution", None)

        if dist:
            metrics["bloom_confidence"].append(float(max(dist)))
        else:
            metrics["bloom_confidence"].append(0.0)

    avg = {k: float(np.mean(v)) for k, v in metrics.items()}

    return avg, metrics


def plot_bar(avg, save_dir):

    names = list(avg.keys())
    values = list(avg.values())

    plt.figure()
    plt.bar(names, values)
    plt.xticks(rotation=45)
    plt.title("Average Metrics")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/avg_metrics.png")
    plt.close()


def plot_distribution(metrics, save_dir):

    for k, values in metrics.items():

        plt.figure()
        plt.hist(values, bins=10)
        plt.title(f"{k} Distribution")
        plt.savefig(f"{save_dir}/{k}_hist.png")
        plt.close()


def main():

    save_dir = find_latest_results()

    print(f"📂 Using results from: {save_dir}")

    results = load_results()

    avg, metrics = aggregate(results)

    print("\n📊 AVERAGE METRICS:")
    for k, v in avg.items():
        print(f"{k}: {v:.4f}")

    plot_bar(avg, save_dir)
    plot_distribution(metrics, save_dir)

    print("\n📈 Graphs saved inside:", save_dir)


if __name__ == "__main__":
    main()