import numpy as np
from PIL import Image

from src.loaders.multimodal_loader import load_image_text
from src.loaders.image_loader import load_image_caption


class ImageProcessor:
    """
    Lightweight multimodal perception:
    OCR + Caption fusion
    """

    def __init__(self):
        pass

    def process(self, image_path: str):
        """
        Returns unified textual representation of image
        """

        # 1. OCR (structure + labels)
        ocr_text = load_image_text(image_path)

        # 2. Caption (semantic understanding)
        caption = load_image_caption(image_path)

        # 3. Fusion
        fused = f"""
[IMAGE CAPTION]
{caption}

[OCR TEXT]
{ocr_text}
""".strip()

        return fused