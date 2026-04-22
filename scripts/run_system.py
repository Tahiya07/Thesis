import argparse

from src.llm import load_rag_model
from src.privacy.learned_privacy import RAGEngine
from src.bloom import classify_bloom
from src.summarizer import summarize
from src.image_pipeline import ImagePipeline


# =========================
# SYSTEM WRAPPER
# =========================

class AcademicSystem:
    def __init__(self, model_path):
        print("🚀 Loading Qwen GGUF model...")
        self.llm = load_rag_model(model_path)

        print("🧠 Initializing RAG engine...")
        self.rag = RAGEngine()

    # ---------------------
    # ASK QUESTION
    # ---------------------
    def ask(self, question: str, use_rag=True):
        if use_rag:
            context = "\n".join(self.rag.retrieve(question))
        else:
            context = ""

        from src.llm import generate

        result = generate(
            self.llm,
            prompt=question,
            context=context if context else None,
            temperature=0.2,
            max_tokens=256
        )

        return {
            "answer": result["response"],
            "context_used": bool(context)
        }

    # ---------------------
    # PDF
    # ---------------------
    def add_pdf(self, path):
        self.rag.build_from_pdf(path)

    # ---------------------
    # URL
    # ---------------------
    def add_url(self, url):
        self.rag.build_from_url(url)

    # ---------------------
    # IMAGE
    # ---------------------
    def ask_image(self, image_path, question):
        print("🖼️ Processing image...")

        pipeline = ImagePipeline()
        result = pipeline.process(image_path)

        # Inject into RAG
        self.rag.build_from_text(result["fused_text"])

        from src.llm import generate

        response = generate(
            self.llm,
            prompt=f"""
    Question: {question}

    IMPORTANT:
    - Image interpretation may be noisy
    - Trust OCR more than caption
    - If uncertain, say "uncertain"

    Answer carefully.
    """,
            context=result["fused_text"],
            temperature=0.2,
            max_tokens=256
        )

        return {
            "answer": response["response"],
            "mode": result["mode"],
            "confidence": result["confidence"]
        }


# =========================
# MAIN CLI
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)

    parser.add_argument("--question", type=str)
    parser.add_argument("--pdf", type=str)
    parser.add_argument("--url", type=str)
    parser.add_argument("--image", type=str)

    args = parser.parse_args()

    system = AcademicSystem(args.model_path)

    # ---------------------
    # PDF MODE
    # ---------------------
    if args.pdf:
        system.add_pdf(args.pdf)
        print("✅ PDF loaded")

    # ---------------------
    # URL MODE
    # ---------------------
    if args.url:
        system.add_url(args.url)
        print("✅ URL loaded")

    # ---------------------
    # IMAGE MODE
    # ---------------------
    if args.image and args.question:
        out = system.ask_image(args.image, args.question)

        print("\n================ ANSWER ================\n")
        print(out["answer"])

        print("\nBloom Level:", classify_bloom(system.llm, args.question))
        print("Used RAG:", out["context_used"])
        return

    # ---------------------
    # TEXT QUESTION MODE
    # ---------------------
    if args.question:
        out = system.ask(args.question)

        print("\n================ ANSWER ================\n")
        print(out["answer"])

        print("\nBloom Level:", classify_bloom(system.llm, args.question))
        print("Used RAG:", out["context_used"])
        return

    print("❌ No valid input provided")


if __name__ == "__main__":
    main()