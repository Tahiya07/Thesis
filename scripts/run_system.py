import gc
import numpy as np

from src.rag_engine import RAGEngine
from src.llm import generate
from src.privacy.privacy_model import build_default_privacy_model


class AcademicSystem:

    def __init__(self, model_path):
        print("System initialized")

        self.model_path = model_path
        self.llm = None
        self.rag = RAGEngine(privacy_model=build_default_privacy_model())

    def get_llm(self):
        if self.llm is None:
            from src.llm import load_rag_model
            self.llm = load_rag_model(self.model_path)
        return self.llm

    def ask(self, question: str, use_rag=True):

        chunks = self.rag.retrieve(question, k=3) if use_rag else []
        context = "\n\n".join([c["text"] for c in chunks]) if chunks else ""
        used_rag = bool(context)

        confidence = float(np.mean([c["score"] for c in chunks])) if chunks else 0.0
        uncertain = confidence < 0.35

        llm = self.get_llm()

        response = generate(
            llm,
            prompt=question,
            context=context if used_rag else None,
            temperature=0.2,
            max_tokens=256
        )

        return {
            "answer": response.get("response", ""),
            "context_used": used_rag,
            "retrieval_score": min(len(chunks) / 3, 1.0),
            "confidence": confidence,
            "uncertain": uncertain,
            "chunks": chunks
        }

    def add_pdf(self, path: str):
        from src.loaders.pdf_loader import load_pdf_text

        text = load_pdf_text(path)
        if text:
            print("Adding PDF...")
            self.rag.add_text(text)
        else:
            print("PDF load failed")

    def add_image(self, image_path: str):
        from src.image_pipeline import ImageProcessor

        print(f"Processing image: {image_path}")
        fused_text = ImageProcessor().process(image_path)
        self.rag.add_text(fused_text)

    def add_url(self, url: str):
        from src.loaders.web_loader import load_webpage

        text = load_webpage(url)
        if text:
            self.rag.add_text(text)


gc.collect()
