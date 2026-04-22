import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# THESIS-GRADE VISUALIZATION DASHBOARD
# =========================================================
def plot_results(csv_path):

    df = pd.read_csv(csv_path)

    # =====================================================
    # 1. BLEU SCORE COMPARISON
    # =====================================================
    plt.figure()
    plt.plot(df["bleu_rag"], label="RAG")
    plt.plot(df["bleu_no_rag"], label="No RAG")
    plt.title("BLEU Score Comparison")
    plt.xlabel("Samples")
    plt.ylabel("BLEU Score")
    plt.legend()
    plt.grid()
    plt.show()

    # =====================================================
    # 2. HALLUCINATION (LLM JUDGE)
    # =====================================================
    plt.figure()
    plt.plot(df["llm_hall_rag"], label="RAG")
    plt.plot(df["llm_hall_no_rag"], label="No RAG")
    plt.title("Hallucination Score (LLM Judge)")
    plt.xlabel("Samples")
    plt.ylabel("Score (0-1)")
    plt.legend()
    plt.grid()
    plt.show()

    # =====================================================
    # 3. LATENCY COMPARISON
    # =====================================================
    plt.figure()
    plt.plot(df["latency_rag"], label="RAG")
    plt.plot(df["latency_no_rag"], label="No RAG")
    plt.title("Latency Comparison")
    plt.xlabel("Samples")
    plt.ylabel("Seconds")
    plt.legend()
    plt.grid()
    plt.show()

    # =====================================================
    # 4. FAITHFULNESS (NEW IMPORTANT THESIS METRIC)
    # =====================================================
    if "faith_rag" in df.columns:

        plt.figure()
        plt.plot(df["faith_rag"], label="RAG Faithfulness")
        plt.plot(df["faith_no_rag"], label="No RAG Faithfulness")
        plt.title("Faithfulness Comparison")
        plt.xlabel("Samples")
        plt.ylabel("Score (0-1)")
        plt.legend()
        plt.grid()
        plt.show()

    # =====================================================
    # 5. RETRIEVAL QUALITY (VERY IMPORTANT FOR RAG THESIS)
    # =====================================================
    if "precision@k" in df.columns and "recall@k" in df.columns:

        plt.figure()
        plt.plot(df["precision@k"], label="Precision@K")
        plt.plot(df["recall@k"], label="Recall@K")
        plt.title("RAG Retrieval Quality")
        plt.xlabel("Samples")
        plt.ylabel("Score")
        plt.legend()
        plt.grid()
        plt.show()

    # =====================================================
    # 6. THESIS SUMMARY BAR CHART (MOST IMPORTANT FIGURE)
    # =====================================================
    metrics = {
        "BLEU (RAG)": df["bleu_rag"].mean(),
        "BLEU (No RAG)": df["bleu_no_rag"].mean(),
        "Hallucination (RAG)": df["llm_hall_rag"].mean(),
        "Hallucination (No RAG)": df["llm_hall_no_rag"].mean(),
        "Latency (RAG)": df["latency_rag"].mean(),
        "Latency (No RAG)": df["latency_no_rag"].mean(),
    }

    plt.figure()
    plt.bar(metrics.keys(), metrics.values())
    plt.title("Overall System Performance Summary")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Value")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # =========================================================
    # PRINT NUMERIC SUMMARY (FOR THESIS TEXT)
    # =========================================================
    print("\n📊 THESIS SUMMARY STATISTICS\n")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")