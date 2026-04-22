import json
import pandas as pd

from scripts.run_system import AcademicSystem
from evaluation.metrics import RAGEvaluator


def run_evaluation(dataset_path, model_path):

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    system = AcademicSystem(model_path)
    evaluator = RAGEvaluator()

    results = []

    for i, sample in enumerate(data):

        question = sample["question"]
        reference = sample.get("reference_answer", "")
        context = sample.get("context", "")
        keywords = sample.get("keywords", [])

        # =========================
        # RETRIEVAL
        # =========================
        retrieved_docs = system.rag.retrieve(question)
        retrieved_text = "\n".join(retrieved_docs)

        # =========================
        # RAG MODE
        # =========================
        out_rag, t_rag = evaluator.measure_latency(system.ask, question, True)

        # =========================
        # NO RAG MODE
        # =========================
        out_no_rag, t_no_rag = evaluator.measure_latency(system.ask, question, False)

        answer_rag = out_rag["answer"]
        answer_no_rag = out_no_rag["answer"]

        # =========================
        # METRICS
        # =========================
        rag_bleu = evaluator.compute_bleu(reference, answer_rag)
        nr_bleu = evaluator.compute_bleu(reference, answer_no_rag)

        rag_hall = evaluator.hallucination_score(retrieved_text, answer_rag)
        nr_hall = evaluator.hallucination_score(context, answer_no_rag)

        rag_faith = evaluator.faithfulness(retrieved_text, answer_rag)
        nr_faith = evaluator.faithfulness(context, answer_no_rag)

        rag_llm = evaluator.llm_hallucination_judge(system.llm, retrieved_text, answer_rag)
        nr_llm = evaluator.llm_hallucination_judge(system.llm, context, answer_no_rag)

        rag_len = evaluator.answer_length_metrics(answer_rag)
        nr_len = evaluator.answer_length_metrics(answer_no_rag)

        precision = evaluator.precision_at_k(retrieved_docs, keywords)
        recall = evaluator.recall_at_k(retrieved_docs, keywords)

        # =========================
        # STORE
        # =========================
        results.append({
            "question": question,

            "bleu_rag": rag_bleu,
            "bleu_no_rag": nr_bleu,

            "hall_rag": rag_hall,
            "hall_no_rag": nr_hall,

            "faith_rag": rag_faith,
            "faith_no_rag": nr_faith,

            "llm_hall_rag": rag_llm,
            "llm_hall_no_rag": nr_llm,

            "latency_rag": t_rag,
            "latency_no_rag": t_no_rag,

            "words_rag": rag_len["word_len"],
            "words_no_rag": nr_len["word_len"],

            "precision@k": precision,
            "recall@k": recall
        })

        print(f"✅ Processed {i+1}/{len(data)}")

    df = pd.DataFrame(results)
    df.to_csv("evaluation/results.csv", index=False)

    print("\n📊 FINAL SUMMARY\n")
    print(df.mean(numeric_only=True))

    return df


if __name__ == "__main__":
    run_evaluation(
        dataset_path="evaluation/dataset.json",
        model_path="models/qwen.gguf"
    )