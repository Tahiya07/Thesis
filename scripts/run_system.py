from src.rag_engine import RAGEngine
from src.llm import generate
from src.ldl import BloomLDL


class AcademicSystem:

    def __init__(self, model_path):
        print("⚡ System initialized")

        self.model_path = model_path
        self.llm = None
        self.rag = RAGEngine()
        self.ldl = BloomLDL()

    # =====================================================
    def get_llm(self):
        if self.llm is None:
            from src.llm import load_rag_model
            self.llm = load_rag_model(self.model_path)
        return self.llm

    # =====================================================
    # ASK (FIXED - STRICT + LOW HALLUCINATION)
    # =====================================================
    def ask(self, question: str, use_rag=True):

        chunks = []
        if use_rag:
            chunks = self.rag.retrieve(question, k=3)

        context = "\n\n".join([c["text"] for c in chunks]) if chunks else ""
        used_rag = len(context) > 0

        uncertain = self.uncertainty_check(chunks)

        llm = self.get_llm()

        # =========================
        # 🔥 FIXED GENERATION FLOW
        # =========================
        if used_rag:

            response = generate(
                llm,
                prompt=question,  # only question here
                context=context,  # ✅ PASS CONTEXT HERE
                temperature=0.2,
                max_tokens=256
            )

        else:
            response = generate(
                llm,
                prompt=question,
                context=None,
                temperature=0.2,
                max_tokens=256
            )

        return {
            "answer": response.get("response", ""),
            "context_used": used_rag,
            "retrieval_score": min(len(chunks) / 3, 1.0),
            "uncertain": uncertain,
            # "chunks": chunks
        }
    def _clean_answer(self, text: str):

        if not text:
            return text

        # remove repeated lines
        lines = list(dict.fromkeys(text.split("\n")))

        text = " ".join(lines)

        # remove extra spaces
        text = " ".join(text.split())

        # cut overly long answers (keeps it tight)
        words = text.split()
        if len(words) > 80:
            text = " ".join(words[:80]) + "..."

        return text
    # =====================================================
    # PDF (FIXED)
    # =====================================================
    def add_pdf(self, path: str):
        from src.loaders.pdf_loader import load_pdf_text

        text = load_pdf_text(path)

        if text:
            print("📄 Adding PDF...")
            self.rag.add_text(text)
        else:
            print("⚠️ PDF load failed")

    # =====================================================
    # IMAGE (FIXED - NO UNKNOWN FILES)
    # =====================================================
    def add_image(self, image_path: str):

        print(f"🖼️ Processing image: {image_path}")

        from src.loaders.multimodal_loader import load_image_text
        from src.blip_captioner import BLIPCaptioner

        # OCR
        ocr_text = load_image_text(image_path)

        # Caption
        captioner = BLIPCaptioner()
        caption = captioner.caption(image_path)

        # 🔥 FUSION
        fused_text = f"{ocr_text}\n{caption}"

        self.rag.add_text(fused_text)

    # =====================================================
    def add_url(self, url: str):
        from src.loaders.web_loader import load_webpage

        text = load_webpage(url)
        if text:
            self.rag.add_text(text)

    # =====================================================
    def uncertainty_check(self, chunks):

        if not chunks:
            return True

        if len(chunks) < 2:
            return True

        avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
        return avg_len < 80

import gc
gc.collect()