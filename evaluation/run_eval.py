import json
import pandas as pd

from scripts.run_system import AcademicSystem
from evaluation.metrics import RAGEvaluator


# =========================================================
# MAIN EVALUATION LOOP (THESIS-GRADE FIXED VERSION)
# =========================================================
def run_evaluation(dataset_path, model_path):

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    system = AcademicSystem(model_path)
    eval = RAGEvaluator()

    results = []

    for i, sample in enumerate(data):

        question = sample["question"]
        reference = sample.get("reference_answer", "")
        context = sample.get("context", "")
        relevant_keywords = sample.get("keywords", [])

        # =====================================================
        # RAG RETRIEVAL (NEW IMPORTANT METRIC INPUT)
        # =====================================================
        retrieved_docs = system.rag.retrieve(question)

        retrieved_text = "\n".join(retrieved_docs)

        # =====================================================
        # RAG MODE
        # =====================================================
        out_rag, t_rag = eval.measure_latency(system.ask, question, True)

        # =====================================================
        # NO-RAG MODE
        # =====================================================
        out_no_rag, t_no_rag = eval.measure_latency(system.ask, question, False)

        answer_rag = out_rag["answer"]
        answer_no_rag = out_no_rag["answer"]

        # =====================================================
        # CORE METRICS (RAG)
        # =====================================================
        rag_bleu = eval.compute_bleu(reference, answer_rag)
        rag_hall = eval.hallucination_score(retrieved_text, answer_rag)
        rag_faith = eval.faithfulness(retrieved_text, answer_rag)
        rag_llm_hall = eval.llm_hallucination_judge(system.llm, retrieved_text, answer_rag)
        rag_len = eval.answer_length_metrics(answer_rag)

        # =====================================================
        # CORE METRICS (NO RAG)
        # =====================================================
        nr_bleu = eval.compute_bleu(reference, answer_no_rag)
        nr_hall = eval.hallucination_score(context, answer_no_rag)
        nr_faith = eval.faithfulness(context, answer_no_rag)
        nr_llm_hall = eval.llm_hallucination_judge(system.llm, context, answer_no_rag)
        nr_len = eval.answer_length_metrics(answer_no_rag)

        # =====================================================
        # RAG RETRIEVAL QUALITY METRICS
        # =====================================================
        precision_k = eval.precision_at_k(retrieved_docs, relevant_keywords)
        recall_k = eval.recall_at_k(retrieved_docs, relevant_keywords)

        # =====================================================
        # STORE RESULTS
        # =====================================================
        results.append({
            "question": question,

            # latency
            "latency_rag": t_rag,
            "latency_no_rag": t_no_rag,

            # BLEU
            "bleu_rag": rag_bleu,
            "bleu_no_rag": nr_bleu,

            # hallucination
            "hall_rag": rag_hall,
            "hall_no_rag": nr_hall,

            # faithfulness
            "faith_rag": rag_faith,
            "faith_no_rag": nr_faith,

            # LLM judge
            "llm_hall_rag": rag_llm_hall,
            "llm_hall_no_rag": nr_llm_hall,

            # length
            "words_rag": rag_len["word_len"],
            "words_no_rag": nr_len["word_len"],

            # retrieval quality (IMPORTANT THESIS METRICS)
            "precision@k": precision_k,
            "recall@k": recall_k
        })

        print(f"✅ Processed {i+1}/{len(data)}")

    # =========================================================
    # SAVE RESULTS
    # =========================================================
    df = pd.DataFrame(results)
    df.to_csv("evaluation/results.csv", index=False)

    # =========================================================
    # SUMMARY (THESIS TABLE OUTPUT)
    # =========================================================
    print("\n📊 FINAL THESIS SUMMARY\n")

    print("BLEU:")
    print(df[["bleu_rag", "bleu_no_rag"]].mean())

    print("\nHALLUCINATION (TF-IDF):")
    print(df[["hall_rag", "hall_no_rag"]].mean())

    print("\nFAITHFULNESS:")
    print(df[["faith_rag", "faith_no_rag"]].mean())

    print("\nLLM HALLUCINATION:")
    print(df[["llm_hall_rag", "llm_hall_no_rag"]].mean())

    print("\nLATENCY:")
    print(df[["latency_rag", "latency_no_rag"]].mean())

    print("\nRETRIEVAL QUALITY:")
    print(df[["precision@k", "recall@k"]].mean())

    return df


# =========================================================
# RUN DIRECTLY
# =========================================================
if __name__ == "__main__":
    run_evaluation(
        dataset_path="evaluation/dataset.json",
        model_path="models/qwen.gguf"
    )