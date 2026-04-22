import re
from typing import Dict

from src.loaders.image_loader import load_image_caption
from src.loaders.multimodal_loader import load_image_text


class ImagePipeline:
    """
    Research-grade multimodal image processor.

    Features:
    - OCR-first strategy (for academic reliability)
    - Caption fallback (semantic understanding)
    - Noise filtering (hallucination reduction)
    - Confidence-aware fusion
    """

    def __init__(self):
        pass

    # -----------------------------
    # GARBAGE / NOISE DETECTION
    # -----------------------------
    def _is_garbage(self, text: str) -> bool:
        if not text or len(text.strip()) < 10:
            return True

        words = text.lower().split()
        unique_ratio = len(set(words)) / (len(words) + 1e-6)

        # Detect repetition hallucination (e.g., "aro aro aro")
        if unique_ratio < 0.3:
            return True

        # Detect non-informative patterns
        if re.fullmatch(r"[a-z\s]+", text.lower()) and len(set(words)) < 3:
            return True

        return False

    # -----------------------------
    # CONFIDENCE HEURISTICS
    # -----------------------------
    def _compute_confidence(self, ocr: str, caption: str) -> Dict:
        ocr_len = len(ocr.strip())
        cap_len = len(caption.strip())

        return {
            "ocr_available": ocr_len > 20,
            "caption_valid": not self._is_garbage(caption),
            "ocr_length": ocr_len,
            "caption_length": cap_len,
        }

    # -----------------------------
    # MAIN PROCESSOR
    # -----------------------------
    def process(self, image_path: str) -> Dict:
        """
        Returns structured multimodal representation
        """

        # 1. Extract modalities
        ocr_text = ""
        caption = ""

        try:
            ocr_text = load_image_text(image_path)
        except Exception:
            ocr_text = ""

        try:
            caption = load_image_caption(image_path)
        except Exception:
            caption = ""

        # 2. Clean caption if garbage
        if self._is_garbage(caption):
            caption = ""

        # 3. Compute confidence signals
        conf = self._compute_confidence(ocr_text, caption)

        # 4. Fusion strategy
        if conf["ocr_available"]:
            mode = "OCR_PRIMARY"
            fused_text = f"""
[IMAGE ANALYSIS MODE: OCR-PRIMARY]

Extracted Text (Reliable):
{ocr_text}

Caption (Supplementary):
{caption}
"""
        elif conf["caption_valid"]:
            mode = "CAPTION_PRIMARY"
            fused_text = f"""
[IMAGE ANALYSIS MODE: CAPTION-PRIMARY]

Description:
{caption}
"""
        else:
            mode = "UNRELIABLE"
            fused_text = """
[IMAGE ANALYSIS MODE: UNRELIABLE]

The image could not be reliably interpreted.
"""

        return {
            "fused_text": fused_text.strip(),
            "ocr": ocr_text,
            "caption": caption,
            "mode": mode,
            "confidence": conf,
        }