import pandas as pd


class LatexReport:
    """
    Generates thesis-ready LaTeX tables from evaluation results.
    """

    def __init__(self, csv_path="evaluation/results.csv"):
        self.df = pd.read_csv(csv_path)

    # =========================================================
    # MAIN SUMMARY TABLE
    # =========================================================
    def generate_main_table(self):
        df = self.df

        table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{RAG vs No-RAG Performance Comparison}}
\\begin{{tabular}}{{lcc}}
\\hline
Metric & RAG & No-RAG \\\\
\\hline
BLEU & {df['bleu_rag'].mean():.3f} & {df['bleu_no_rag'].mean():.3f} \\\\
Hallucination (LLM) & {df['llm_hall_rag'].mean():.3f} & {df['llm_hall_no_rag'].mean():.3f} \\\\
Latency (s) & {df['latency_rag'].mean():.3f} & {df['latency_no_rag'].mean():.3f} \\\\
Faithfulness & {df['faith_rag'].mean():.3f} & {df['faith_no_rag'].mean():.3f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""

        with open("evaluation/results_table.tex", "w", encoding="utf-8") as f:
            f.write(table)

        print("📄 LaTeX table saved → evaluation/results_table.tex")

    # =========================================================
    # RETRIEVAL TABLE
    # =========================================================
    def generate_retrieval_table(self):
        if "precision@k" not in self.df.columns:
            print("⚠️ Retrieval metrics not found")
            return

        table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{RAG Retrieval Performance}}
\\begin{{tabular}}{{lc}}
\\hline
Metric & Score \\\\
\\hline
Precision@K & {self.df['precision@k'].mean():.3f} \\\\
Recall@K & {self.df['recall@k'].mean():.3f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""

        with open("evaluation/retrieval_table.tex", "w", encoding="utf-8") as f:
            f.write(table)

        print("📄 Retrieval LaTeX saved → evaluation/retrieval_table.tex")