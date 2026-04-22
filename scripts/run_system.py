import argparse
import os

from src.llm import load_rag_model, generate, classify_bloom
from src.rag_engine import RAGEngine
from src.rag_token_manager import TokenManager


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

        # 🔥 TOKEN MANAGER (NEW)
        self.token_manager = TokenManager(model_context_size=2304)

    # -------------------------
    # TEXT QUESTION
    # -------------------------
    def ask(self, question: str, use_rag=True):

        raw_context = self.rag.retrieve(question) if use_rag else []

        # 🔥 SAFE CONTEXT BUILDING
        context_text = self.token_manager.build_safe_context(
            raw_context,
            question
        )

        response = generate(
            self.llm,
            prompt=question,
            context=context_text if context_text else None,
            temperature=0.2,
            max_tokens=256
        )

        return {
            "answer": response["response"],
            "context_used": bool(raw_context),
            "safe_context": True
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
        return

    print("❌ No valid input provided.")


if __name__ == "__main__":
    main()