import argparse
import os
import numpy as np

from src.llm import load_rag_model, generate, classify_bloom
from src.rag_engine import RAGEngine
from src.rag_token_manager import TokenManager
from sentence_transformers import SentenceTransformer


# =========================================================
# SYSTEM WRAPPER
# =========================================================
class AcademicSystem:
    def __init__(self, model_path):
        print("⚡ System booting (lazy mode)...")

        print("🧠 Loading LLM (first use only)...")
        self.llm = load_rag_model(model_path)

        print("📚 Initializing RAG (first use only)...")
        self.rag = RAGEngine()

        # token safety
        self.token_manager = TokenManager(model_context_size=2304)

        # =====================================================
        # EMBEDDING MODEL (USED FOR RAG GATING)
        # =====================================================
        print("🧠 Loading embedding model (RAG scoring)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # -------------------------
    # COSINE SIMILARITY
    # -------------------------
    def _cosine(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    # -------------------------
    # RETRIEVAL SCORING (FIXED)
    # -------------------------
    def retrieval_score(self, context: str, question: str):
        if not context or not question:
            return 0.0

        c_emb = self.embedder.encode(context)
        q_emb = self.embedder.encode(question)

        return self._cosine(c_emb, q_emb)

    # -------------------------
    # TEXT QUESTION
    # -------------------------
    def ask(self, question: str, use_rag=True):

        raw_context = []
        context_text = ""
        used_rag = False

        # =========================
        # STEP 1: RETRIEVE
        # =========================
        if use_rag:
            raw_context = self.rag.retrieve(question)

        # =========================
        # STEP 2: RAG GATING (FIXED)
        # =========================
        if raw_context:
            combined_context = "\n".join(raw_context)

            score = self.retrieval_score(combined_context, question)

            # ✔ stable semantic threshold
            if score > 0.35:
                context_text = self.token_manager.build_safe_context(
                    raw_context,
                    question
                )
                used_rag = True
            else:
                context_text = ""
                used_rag = False

        # =========================
        # STEP 3: PROMPT CONSTRUCTION
        # =========================
        if used_rag:
            prompt = f"""
You MUST answer using ONLY the provided context.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""
        else:
            prompt = f"""
Answer the question clearly and concisely.

QUESTION:
{question}

ANSWER:
"""

        # =========================
        # STEP 4: GENERATION
        # =========================
        response = generate(
            self.llm,
            prompt=prompt,
            temperature=0.2,
            max_tokens=256
        )

        return {
            "answer": response["response"],
            "context_used": used_rag,
            "retrieval_score": score if raw_context else 0.0
        }

    # -------------------------
    # PDF INGESTION
    # -------------------------
    def add_pdf(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")

        self.rag.add_pdf(path)
        print("✅ PDF loaded into RAG")

    # -------------------------
    # URL INGESTION
    # -------------------------
    def add_url(self, url):
        self.rag.add_url(url)
        print("✅ URL loaded into RAG")


# =========================================================
# MAIN CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--question", type=str)
    parser.add_argument("--pdf", type=str)
    parser.add_argument("--url", type=str)

    args = parser.parse_args()

    system = AcademicSystem(args.model_path)

    # -------------------------
    # INGESTION
    # -------------------------
    if args.pdf:
        system.add_pdf(args.pdf)

    if args.url:
        system.add_url(args.url)

    # -------------------------
    # QUESTION
    # -------------------------
    if args.question:
        out = system.ask(args.question)

        print("\n================ ANSWER ================\n")
        print(out["answer"])

        try:
            bloom = classify_bloom(system.llm, args.question)
        except:
            bloom = "Unknown"

        print("\nBloom Level:", bloom)
        print("Used RAG:", out["context_used"])
        print("Retrieval Score:", round(out["retrieval_score"], 4))
        return

    print("❌ No valid input provided.")


if __name__ == "__main__":
    main()